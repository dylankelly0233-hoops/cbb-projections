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
# We use st.secrets for security so your API key isn't public on GitHub
# If running locally, you can hardcode it, but for web, use secrets (explained below).
try:
    API_KEY = st.secrets["API_KEY"]
except:
    API_KEY = 'rTQCNjitVG9Rs6LDYzuUVU4YbcpyVCA6mq2QSkPj8iTkxi3UBVbic+obsBlk7JCo'  # Fallback for local testing

YEAR = 2026
DECAY_ALPHA = 0.035
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


# We use @st.cache_data so it doesn't spam the API every time you click a button.
# It refreshes data once every hour (3600 seconds).
@st.cache_data(ttl=3600)
def fetch_api_data(year):
    with st.spinner('Fetching latest data from API...'):
        # Fetch Games
        games_resp = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={'season': year})
        games = games_resp.json() if games_resp.status_code == 200 else []

        # Fetch Lines
        lines_resp = requests.get(f"{BASE_URL}/lines", headers=HEADERS, params={'season': year})
        lines = lines_resp.json() if lines_resp.status_code == 200 else []

        return games, lines


# --- MAIN LOGIC ---

def run_analysis():
    st.title(f"🏀 CBB Power Ratings & Projections {YEAR}")

    # 1. Fetch Data
    games_json, lines_json = fetch_api_data(YEAR)

    if not games_json:
        st.error("Failed to load games data.")
        return

    # Process Dates
    now_et = datetime.now(timezone(timedelta(hours=-5)))
    today_str = now_et.strftime('%Y-%m-%d')
    st.caption(f"Last Updated: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")

    # 2. Process Games
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

    # 3. Build Training Matrix
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

        if days_ago < 0: continue  # Skip future

        weight = np.exp(-DECAY_ALPHA * days_ago)
        matchups.append({
            'Home': home, 'Away': away, 'Margin': -1 * float(s),
            'Is_Neutral': meta['is_neutral'], 'Weight': weight
        })

    df = pd.DataFrame(matchups)

    if df.empty:
        st.warning("No past line data found to train model.")
        return

    # 4. Regression
    home_dummies = pd.get_dummies(df['Home'], dtype=int)
    away_dummies = pd.get_dummies(df['Away'], dtype=int)
    all_teams = sorted(list(set(home_dummies.columns) | set(away_dummies.columns)))

    home_dummies = home_dummies.reindex(columns=all_teams, fill_value=0)
    away_dummies = away_dummies.reindex(columns=all_teams, fill_value=0)

    X = home_dummies.sub(away_dummies)
    X['HFA_Constant'] = df['Is_Neutral'].apply(lambda x: 0 if x else 1)
    y = df['Margin']
    w_vals = df['Weight'].values
    w_norm = w_vals * (len(w_vals) / w_vals.sum())

    clf = Ridge(alpha=0.001, fit_intercept=False)
    clf.fit(X, y, sample_weight=w_norm)

    coefs = pd.Series(clf.coef_, index=X.columns)
    implied_hca = coefs['HFA_Constant']
    market_ratings = coefs.drop('HFA_Constant') - coefs.drop('HFA_Constant').mean()

    # 5. Display Ratings
    ratings_df = pd.DataFrame({'Team': market_ratings.index, 'Rating': market_ratings.values})
    ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
    ratings_df.index += 1

    st.sidebar.header("Market Settings")
    st.sidebar.metric("Implied HCA", f"{implied_hca:.2f}")
    st.sidebar.info(f"Using Decay Alpha: {DECAY_ALPHA}")

    with st.expander("📊 View Power Ratings"):
        st.dataframe(ratings_df, height=300, use_container_width=True)

    # 6. Projections
    st.subheader(f"Games for {today_str}")

    if not todays_games:
        st.info("No games scheduled for today.")
    else:
        todays_games.sort(key=lambda x: x['et_datetime'])
        projections = []

        for g in todays_games:
            h = get_team_name(g.get('homeTeam'))
            a = get_team_name(g.get('awayTeam'))

            h_r = market_ratings.get(h, 0.0)
            a_r = market_ratings.get(a, 0.0)
            is_neutral = g.get('neutralSite', False)
            hca_val = 0.0 if is_neutral else implied_hca

            my_line = (h_r - a_r) + hca_val

            # Find Vegas line
            gid = g.get('id')
            vegas = None
            for l in lines_json:
                if str(l.get('gameId')) == str(gid) and l.get('lines'):
                    s = l['lines'][0].get('spread')
                    if s is not None: vegas = float(s)
                    break

            edge = (my_line - (-vegas)) if vegas is not None else 0

            # Formatting for display
            pick = ""
            if edge > 3.0:
                pick = f"BET {h}"
            elif edge < -3.0:
                pick = f"BET {a}"

            projections.append({
                'Time': g['et_datetime'].strftime('%I:%M %p'),
                'Matchup': f"{a} @ {h}",
                'My Line': round(my_line, 1),
                'Vegas': vegas if vegas is not None else "N/A",
                'Edge': round(edge, 1),
                'Pick': pick
            })

        proj_df = pd.DataFrame(projections)

        # Style the dataframe (Highlight Edges)
        def highlight_picks(val):
            color = ''
            if 'BET' in str(val):
                color = 'background-color: #90EE90; color: black; font-weight: bold'  # Light Green
            return color

        st.dataframe(proj_df.style.applymap(highlight_picks, subset=['Pick']), use_container_width=True)

        # 7. Excel Download
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
