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
    
    # 3. HCA Logic Toggle
    st.sidebar.subheader("🏟️ Home Court Logic")
    use_dynamic_hca = st.sidebar.checkbox("Use Team-Specific HCA", value=False, help="If checked, the model calculates a unique Home Court Advantage for every team.")

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
        
        matchups.append({
            'Home': home, 'Away': away, 'Margin': -1 * float(s),
            'Is_Neutral': meta['is_neutral'], 'Weight': weight
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

    X_ratings = home_dummies.sub(away_dummies)

    if use_dynamic_hca:
        X_hca = home_dummies.copy()
        is_neutral_mask = df['Is_Neutral'].values
        X_hca.loc[is_neutral_mask, :] = 0
        X_hca.columns = [f"{c}_HCA" for c in X_hca.columns]
        X = pd.concat([X_ratings, X_hca], axis=1)
    else:
        X = X_ratings.copy()
        X['HFA_Constant'] = df['Is_Neutral'].apply(lambda x: 0 if x else 1)

    y = df['Margin']
    w_vals = df['Weight'].values
    w_norm = w_vals * (len(w_vals) / w_vals.sum())

    clf = Ridge(alpha=1.0, fit_intercept=False)
    clf.fit(X, y, sample_weight=w_norm)

    # --- 5. EXTRACT RATINGS ---
    coefs = pd.Series(clf.coef_, index=X.columns)
    
    if use_dynamic_hca:
        rating_cols = [c for c in coefs.index if not c.endswith('_HCA')]
        hca_cols = [c for c in coefs.index if c.endswith('_HCA')]
        
        raw_ratings = coefs[rating_cols]
        hca_vals = coefs[hca_cols]
        hca_vals.index = [c.replace('_HCA', '') for c in hca_vals.index]
        
        # --- GUARDRAILS APPLIED HERE ---
        # 1. Floor at 0.0 (No negative HCA)
        # 2. Ceiling at 6.0 (Max realistic HCA)
        hca_vals = hca_vals.clip(lower=0.0, upper=6.0)
        
        market_ratings = raw_ratings - raw_ratings.mean()
        avg_hca = hca_vals.mean()
        st.sidebar.info(f"Avg HCA (Clipped): {avg_hca:.2f} pts")
        st.sidebar.caption("Max HCA capped at 6.0")
        
    else:
        implied_hca = coefs['HFA_Constant']
        market_ratings = coefs.drop('HFA_Constant') - coefs.drop('HFA_Constant').mean()
        st.sidebar.info(f"Global Fixed HCA: {implied_hca:.2f} pts")

    # Display Top 25
    ratings_df = pd.DataFrame({'Team': market_ratings.index, 'Rating': market_ratings.values})
    ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
    ratings_df.index += 1
    
    with st.expander("📊 View Power Ratings"):
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

            if is_neutral:
                hca_val = 0.0
            else:
                if use_dynamic_hca:
                    hca_val = hca_vals.get(h, avg_hca)
                else:
                    hca_val = implied_hca

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
