import pandas as pd
import joblib
import os
from api import fetch_upcoming_fixtures
from features import build_features
from config import MODEL_DIR, COMPETITIONS
from score_model import load_latest_score_model, predict_score


def load_latest_model():
    """Load the most recent trained model with metadata."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory {MODEL_DIR} not found. Run training first."
        )

    files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError("No model files found. Run training first.")

    # Find the latest model (newest timestamp in filename)
    latest_model_file = files[-1]
    model_path = os.path.join(MODEL_DIR, latest_model_file)

    model_data = joblib.load(model_path)
    print(f"📂 Loaded model from {model_path}")
    print(f"   Trained on: {model_data.get('trained_on', 'unknown')}")
    print(f"   Features used: {model_data.get('features', [])[:5]}...")
    return model_data


def get_prediction_label(prediction):
    """Converts the model's numerical prediction to a human-readable string."""
    if prediction == 0:
        return "Home Win"
    elif prediction == 1:
        return "Draw"
    elif prediction == 2:
        return "Away Win"
    return "Unknown"


def predict_upcoming_matches(limit=10):
    """
    Loads the latest result model, generates features for upcoming fixtures,
    and returns predictions with probabilities.
    """
    try:
        model_data = load_latest_model()
        model = model_data["model"]
        feature_cols = model_data["features"]
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure a model has been trained by running model.py")
        return []

    # 1. Load the master historical dataset
    # This ensures we have the full context for feature calculation
    print("🌍 Loading historical master dataset...")
    try:
        historical_df = pd.read_parquet("data/master_dataset.parquet")
    except FileNotFoundError:
        print("❌ Master dataset 'data/master_dataset.parquet' not found.")
        print("Please run data_integration.py first to create it.")
        return []

    # 2. Fetch upcoming fixtures
    print(f"🔍 Fetching {limit} upcoming fixtures...")
    upcoming_fixtures_raw = fetch_upcoming_fixtures(limit=limit)
    if not upcoming_fixtures_raw:
        print("No upcoming fixtures found.")
        return []

    # 3. Prepare upcoming fixtures as a DataFrame, matching historical_df schema
    upcoming_records = []
    for f in upcoming_fixtures_raw:
        upcoming_records.append(
            {
                "date": pd.to_datetime(f["date"]),
                "home_team": f["home"],
                "away_team": f["away"],
                "fthg": None,  # Future matches have no full time goals
                "ftag": None,
                "ftr": None,  # No result yet
                "hthg": None,  # Future matches have no half time goals
                "htag": None,
                "b365h": f.get("odds_h"),  # Use available odds
                "b365d": f.get("odds_d"),
                "b365a": f.get("odds_a"),
            }
        )
    upcoming_df = pd.DataFrame(upcoming_records)
    # Add season column, defaulting to current year if not specified by API or for upcoming
    upcoming_df["season"] = upcoming_df["date"].dt.year.astype(str)

    # Ensure all columns exist, fill missing with None/NaN as appropriate
    # This is important to avoid errors when concatenating with historical_df
    common_cols = list(set(historical_df.columns) | set(upcoming_df.columns))
    for col in common_cols:
        if col not in historical_df.columns:
            historical_df[col] = None
        if col not in upcoming_df.columns:
            upcoming_df[col] = None

    # Sort columns to ensure consistent order before concat (important for some pandas versions)
    historical_df = historical_df[sorted(historical_df.columns)]
    upcoming_df = upcoming_df[sorted(upcoming_df.columns)]

    # 4. Combine historical data with upcoming fixtures
    # This combined DataFrame will be used to build features, as rolling calculations
    # need historical context right up to the point of the upcoming game.
    combined_df = pd.concat([historical_df, upcoming_df], ignore_index=True)
    combined_df = combined_df.sort_values(by="date").reset_index(drop=True)

    # 5. Build features for the combined dataset
    # We set drop_na_labels=False to keep upcoming matches (which have NaN labels)
    print("📊 Building features for upcoming matches...")
    features_all_data = build_features(
        df=combined_df, drop_na_labels=False  # Pass the combined DataFrame directly
    )

    # 6. Filter for upcoming matches from the feature matrix
    upcoming_features_df = features_all_data[features_all_data["label"].isna()].copy()

    if upcoming_features_df.empty:
        print(
            "No features generated for upcoming fixtures. Check data or feature pipeline."
        )
        return []

    # 7. Prepare feature matrix X for prediction
    # Ensure the feature columns are exactly what the model was trained on
    X_predict = upcoming_features_df[feature_cols]

    # 8. Make predictions
    probabilities = model.predict_proba(X_predict)
    predictions = model.predict(X_predict)

    # Load score model once outside the loop for efficiency
    score_model_data = None
    try:
        score_model_data = load_latest_score_model()
    except FileNotFoundError:
        print("⚠️ Score model not found. Score predictions will not be available.")
    except Exception as e:
        print(
            f"⚠️ Error loading score model: {e}. Score predictions will not be available."
        )

    # 9. Format results
    results = []
    for i, (idx, row) in enumerate(upcoming_features_df.iterrows()):
        original_fixture_index = upcoming_features_df.index.get_loc(idx)
        original_fixture_data = upcoming_fixtures_raw[original_fixture_index]

        score_pred_str = "N/A (Score Model Update Needed)"
        if score_model_data:
            try:
                # predict_score expects a single-row DataFrame for the features
                score_pred_str = predict_score(
                    score_model_data,
                    upcoming_features_df.loc[[idx]],  # Pass the single-row DataFrame
                )
            except Exception as score_e:
                score_pred_str = f"Score Error: {score_e}"

        results.append(
            {
                "date": original_fixture_data["date"],
                "competition": COMPETITIONS.get(
                    original_fixture_data["competition"],
                    original_fixture_data["competition"],
                ),
                "home": original_fixture_data["home"],
                "away": original_fixture_data["away"],
                "prediction": get_prediction_label(predictions[i]),
                "probabilities": {
                    "home_win": probabilities[i][0],
                    "draw": probabilities[i][1],
                    "away_win": probabilities[i][2],
                },
                "score_prediction": score_pred_str,
            }
        )
    return results


if __name__ == "__main__":
    # Example usage:
    # This will attempt to predict the next 5 upcoming matches
    print("--- Running Upcoming Match Predictor ---")
    predictions = predict_upcoming_matches(limit=5)
    if predictions:
        for p in predictions:
            print(f"\nMatch: {p['home']} vs {p['away']}")
            print(f"Date: {p['date']}")
            print(f"Predicted Outcome: {p['prediction']}")
            print(
                f"Probabilities: Home Win={p['probabilities']['home_win']:.2f}, "
                f"Draw={p['probabilities']['draw']:.2f}, "
                f"Away Win={p['probabilities']['away_win']:.2f}"
            )
            print(f"Score Prediction: {p['score_prediction']}")
    else:
        print("No predictions generated.")
