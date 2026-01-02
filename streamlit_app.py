import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.linear_model import Ridge
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="CBB Projections", layout="wide")

# --- CONFIGURATION ---
API_KEY = 'rTQCNjitVG9Rs6LDYzuUVU4YbcpyVCA6mq2QSkPj8iTkxi3UBVbic+obsBlk7JCo'
YEAR = 2026
BASE_URL = 'https://api.collegebasketballdata.com'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'accept': 'application/json'}

# --- HELPER FUNCTIONS ---
def get_team_name(team_obj):
    if isinstance(team_obj, dict): return team_obj.get('name', 'Unknown')
    return str(team_obj)

def utc_to_et(iso_date_str):
    if not iso_date_str: return datetime.now()
    try:
        dt_utc = datetime.fromisoformat(iso_date_str.replace('Z', '+00:00'))
        dt_et = dt_utc.astimezone(timezone(timedelta(hours=-5)))
        return dt_et
    except ValueError:
        return datetime.now()

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_api_data(year):
    masked_key = API_KEY[:5] + "..." + API_KEY[-5:] if API_KEY else "None"
    
    with st.spinner(f'Fetching data for season {year}...'):
        try:
            games_resp = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={'season': year})
            if games_resp.status_code != 200:
                st.error(f"❌ API Error (Games): {games_resp.status_code}")
                return [], []
            games = games_resp.json()
        except Exception as e:
            st.error(f"❌ Connection Error (Games): {e}")
            return [], []

        try:
            lines_resp = requests.get(f"{BASE_URL}/lines", headers=HEADERS, params={'season': year})
            if lines_resp.status_code != 200:
                st.error(f"❌ API Error (Lines): {lines_resp.status_code}")
                return [], []
            lines = lines_resp.json()
        except Exception as e:
            st.error(f"❌ Connection Error (Lines): {e}")
            return [], []
        
        return games, lines

# --- MAIN LOGIC ---
def run_analysis():
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("⚙️ Settings")
    
    # 1. Date Selector
    st.sidebar.subheader("📅 Date Selection")
    now_et = datetime.now(timezone(timedelta(hours=-5)))
    selected_date = st.sidebar.date_input("Target Date", now_et)
    today_str = selected_date.strftime('%Y-%m-%d')

    # 2. Decay Setting
    decay_alpha = st.sidebar.slider(
        "Decay Rate (Alpha)", 
        min_value=0.000, 
        max_value=0.100, 
        value=0.035, 
        step=0.001,
        format="%.3f"
    )
    
    # 3. MANUAL HCA CONTROL (Replaces the "Dynamic/Regression" logic)
    st.sidebar.subheader("🏟️ Home Court Advantage")
    st.sidebar.caption("Instead of asking the model to guess HCA (which hurts team ratings), set the standard HCA points here.")
    manual_hca = st.sidebar.slider("Points for HCA", 2.0, 5.0, 3.2, 0.1)

    st.title(f"🏀 CBB Projections: {today_str}")

    # --- 1. FETCH DATA ---
    games_json, lines_json = fetch_api_data(YEAR)
    
    if not games_json:
        st.warning("No data loaded.")
        return

    # --- 2. PROCESS GAMES ---
    todays_games = []
    game_meta = {}

    for g in games_json:
        h = get_team_name(g.get('homeTeam'))
        a = get_team_name(g.get('awayTeam'))
        raw_start = g.get('startDate', '')
        dt_et = utc_to_et(raw_start)
        date_et = dt_et.strftime('%Y-%m-%d')

        game_meta[f"{h}_{a}"] = {'is_neutral': g.get('neutralSite', False), 'date_et': date_et}

        if date_et == today_str:
            g['et_datetime'] = dt_et
            todays_games.append(g)

    # --- 3. BUILD TRAINING MATRIX ---
    matchups = []
    target_dt_obj = datetime.strptime(today_str, '%Y-%m-%d')

    for game in lines_json:
        lines = game.get('lines', [])
        if not lines: continue
        s = lines[0].get('spread')
        if s is None: continue

        home = get_team_name(game.get('homeTeam'))
        away = get_team_name(game.get('awayTeam'))
        meta = game_meta.get(f"{home}_{away}")
        
        if not meta: continue

        g_date_obj = datetime.strptime(meta['date_et'], '%Y-%m-%d')
        days_ago = (target_dt_obj - g_date_obj).days

        if days_ago < 0: continue 

        weight = np.exp(-decay_alpha * days_ago)
        
        # --- THE FIX: PRE-BAKE THE HCA ---
        # We adjust the margin BEFORE the model sees it.
        # If Home wins by 10, and HCA is 3.2, we tell the model: "They won by 6.8 on a neutral court."
        # This forces the Team Rating to account for that 6.8 points of skill.
        adjusted_margin = -1 * float(s) # Start with actual margin
        
        if not meta['is_neutral']:
            # Subtract the manual HCA from the home team's margin
            adjusted_margin -= manual_hca
        
        matchups.append({
            'Home': home, 'Away': away, 
            'Adjusted_Margin': adjusted_margin, # Uses adjusted margin
            'Weight': weight
        })

    df = pd.DataFrame(matchups)
    
    if df.empty:
        st.error(f"No past games found prior to {today_str} to train the model.")
        return

    # --- 4. REGRESSION ---
    home_dummies = pd.get_dummies(df['Home'], dtype=int)
    away_dummies = pd.get_dummies(df['Away'], dtype=int)
    all_teams = sorted(list(set(home_dummies.columns) | set(away_dummies.columns)))

    home_dummies = home_dummies.reindex(columns=all_teams, fill_value=0)
    away_dummies = away_dummies.reindex(columns=all_teams, fill_value=0)

    # Simple Matrix: Home - Away
    X = home_dummies.sub(away_dummies)
    
    # We NO LONGER solve for HCA here. We solved it manually above.
    y = df['Adjusted_Margin']
    
    w_vals = df['Weight'].values
    w_norm = w_vals * (len(w_vals) / w_vals.sum())

    clf = Ridge(alpha=1.0, fit_intercept=False) # Fit intercept false because we centered data via HCA subtraction
    clf.fit(X, y, sample_weight=w_norm)

    # --- 5. EXTRACT RATINGS ---
    coefs = pd.Series(clf.coef_, index=X.columns)
    market_ratings = coefs - coefs.mean() # Center around 0

    # Display Top 25
    ratings_df = pd.DataFrame({'Team': market_ratings.index, 'Rating': market_ratings.values})
    ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
    ratings_df.index += 1
    
    with st.expander("📊 View Power Ratings (Pure Skill)"):
        st.dataframe(ratings_df.head(25), height=300, use_container_width=True)

    # --- 6. PROJECTIONS ---
    st.subheader(f"Projections")
    
    if not todays_games:
        st.info(f"No games scheduled for {today_str}.")
    else:
        todays_games.sort(key=lambda x: x['et_datetime'])
        projections = []

        for g in todays_games:
            h = get_team_name(g.get('homeTeam'))
            a = get_team_name(g.get('awayTeam'))
            
            h_r = market_ratings.get(h, 0.0)
            a_r = market_ratings.get(a, 0.0)
            is_neutral = g.get('neutralSite', False)

            # Apply Manual HCA for projection
            hca_val = 0.0 if is_neutral else manual_hca

            # Raw margin (Positive = Home Wins by X)
            raw_margin = (h_r - a_r) + hca_val
            my_proj_spread = -1 * raw_margin

            gid = g.get('id')
            vegas = None
            for l in lines_json:
                if str(l.get('gameId')) == str(gid) and l.get('lines'):
                    s = l['lines'][0].get('spread')
                    if s is not None: vegas = float(s)
                    break
            
            edge = 0.0
            if vegas is not None:
                edge = vegas - my_proj_spread

            pick = ""
            if edge > 3.0: pick = f"BET {h}"
            elif edge < -3.0: pick = f"BET {a}"
            
            projections.append({
                'Time': g['et_datetime'].strftime('%I:%M %p'),
                'Matchup': f"{a} @ {h}",
                'Home Rtg': round(h_r, 1),
                'Away Rtg': round(a_r, 1),
                'My Spread': round(my_proj_spread, 1),
                'Vegas': vegas if vegas is not None else "N/A",
                'Edge': round(edge, 1),
                'Pick': pick,
                'HCA Used': round(hca_val, 1)
            })
            
        proj_df = pd.DataFrame(projections)

        def highlight_picks(val):
            color = ''
            if 'BET' in str(val):
                color = 'background-color: #90EE90; color: black; font-weight: bold'
            return color

        st.dataframe(proj_df.style.applymap(highlight_picks, subset=['Pick']), use_container_width=True)

        # Excel Export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            ratings_df.to_excel(writer, sheet_name='Ratings')
            proj_df.to_excel(writer, sheet_name='Projections', index=False)
        
        st.download_button(
            label="📥 Download Excel Report",
            data=buffer.getvalue(),
            file_name=f"CBB_Projections_{today_str}.xlsx",
            mime="application/vnd.ms-excel"
        )

if __name__ == "__main__":
    run_analysis()
