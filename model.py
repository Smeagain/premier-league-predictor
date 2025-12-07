import os
import joblib
import pandas as pd
from datetime import datetime
from config import MODEL_DIR, MODEL_NAME_TEMPLATE, COMPETITIONS
from api import fetch_all_data, fetch_upcoming_fixtures
from sklearn.ensemble import RandomForestClassifier


def prepare_features(rows):
    """
    Convert list of standings rows into feature matrix X and target vector y.
    """
    df = pd.DataFrame(rows)
    # features
    feature_cols = ['position', 'playedGames', 'goalsFor', 'goalsAgainst']
    X = df[feature_cols]
    # target: did this team finish in top half?
    df['target'] = (df['position'] <= (df.shape[0] // 2)).astype(int)
    y = df['target']

    return X, y


def train_model():
    # Train on the most recent completed season only
    current_year = datetime.now().year
    # season param: use previous calendar year (e.g., 2024 for 2024/25)
    season = current_year - 1
    print(f"🔍 Fetching data for season {season}...")
    rows = fetch_all_data([season])
    if not rows:
        print("No data fetched for the current season; aborting training.")
        return

    X, y = prepare_features(rows)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    path = os.path.join(MODEL_DIR, MODEL_NAME_TEMPLATE.format(timestamp=timestamp))
    joblib.dump(model, path)
    print(f"💾 Model saved to {path}")


def load_latest_model():
    files = sorted(os.listdir(MODEL_DIR))
    if not files:
        raise FileNotFoundError("No model files found. Train first.")
    return joblib.load(os.path.join(MODEL_DIR, files[-1]))


def predict(limit=10):
    """
    Load the latest model, fetch upcoming fixtures, and list them with dates.
    """

    fixtures = fetch_upcoming_fixtures(limit=limit)
    if not fixtures:
        print("No upcoming fixtures found.")
        return

    print(f"\nNext {len(fixtures)} fixtures:\n")
    for m in fixtures:
        dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        comp_name = COMPETITIONS.get(m["competition"], m["competition"])
        print(f"{date_str} — {comp_name}: {m['home']} vs {m['away']}")
    print()
