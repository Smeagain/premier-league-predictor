import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, MODEL_NAME_TEMPLATE
from api import fetch_all_data
from features import build_features


def train_model():
    """
    Train the RandomForest model and persist with versioned filename.
    """
    print("🔍 Fetching data...")
    raw = fetch_all_data()
    df = build_features(raw)

    # Drop rows with missing positions
    df = df.dropna(subset=['home_pos', 'away_pos'])

    X = df.drop(columns=['label'])
    y = df['label']

    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
    ])
    print("🧠 Training model...")
    pipeline.fit(X, y)

    date = datetime.now().strftime('%Y%m%d')
    filename = MODEL_NAME_TEMPLATE.format(date=date)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(pipeline, path)
    print(f"💾 Model saved to {path} ({len(df)} matches)")
    return path


def load_latest_model():
    """
    Load the most recently saved model.
    """
    files = sorted(os.listdir(MODEL_DIR))
    if not files:
        raise FileNotFoundError("No model files found. Train first.")
    latest = files[-1]
    return joblib.load(os.path.join(MODEL_DIR, latest))


def predict(limit=10, as_json=False):
    """
    Predict outcomes for upcoming PL fixtures.
    """
    model = load_latest_model()
    from api import fetch_matches
    from config import COMPETITION_PL
    matches = fetch_matches(COMPETITION_PL, datetime.now().year, status="SCHEDULED")
    fixtures = sorted(matches, key=lambda m: m['utcDate'])[:limit]

    raw = fixtures
    df = build_features(raw)
    preds = model.predict_proba(df.drop(columns=['label']))

    results = []
    for m, p in zip(raw, preds):
        home = m['homeTeam']['name']
        away = m['awayTeam']['name']
        date = m['utcDate']
        results.append({
            'home': home,
            'away': away,
            'date': date,
            'prob': {'home_win': p[0], 'draw': p[1], 'away_win': p[2]}
        })

    if as_json:
        import json
        print(json.dumps(results, indent=2))
        return results
    else:
        for r in results:
            print(f"⚽ {r['home']} vs {r['away']} @ {r['date']}")
            print(f"  Home: {r['prob']['home_win']*100:.1f}%")
            print(f"  Draw: {r['prob']['draw']*100:.1f}%")
            print(f"  Away: {r['prob']['away_win']*100:.1f}%\n")
        return results
