import os
from features import build_features
import statsmodels.api as sm
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import poisson


SCORE_MODEL_DIR = "score_models"
SCORE_MODEL_NAME_TEMPLATE = "score_model_{timestamp}.pkl"


def train_score_model():
    """
    Trains two Poisson Regression models: one for home goals and one for away goals.
    The models use the rich feature set from build_features.
    """
    print("📊 Building feature matrix for score model training...")
    # 1. Build features using our new pipeline
    df = build_features(
        drop_na_labels=True
    )  # drop NA labels to train only on completed matches

    if df.empty:
        print("❌ Feature matrix is empty; aborting score model training.")
        return None

    # Filter out matches where goals are missing (shouldn't happen with drop_na_labels=True, but safe)
    df = df[df["fthg"].notna() & df["ftag"].notna()].copy()

    if df.empty:
        print(
            "❌ No complete match data with goals available; aborting score model training."
        )
        return None

    print(f"✓ Training data for score model: {len(df)} matches")

    # Store all unique team names for dummy variable consistency in prediction
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    # Convert teams to dummy variables across the entire DataFrame
    home_dummies = pd.get_dummies(df["home_team"], prefix="home_team", dtype=int)
    away_dummies = pd.get_dummies(df["away_team"], prefix="away_team", dtype=int)
    df = pd.concat([df, home_dummies, away_dummies], axis=1)

    # Define the core numerical features (excluding home/away team names, dates, labels, etc.)
    core_numerical_features = [
        "elo_diff",
        "home_avg_pts",
        "home_avg_goals_scored",
        "home_avg_goals_conceded",
        "home_avg_goal_diff",
        "home_avg_days_since_last_match",
        "away_avg_pts",
        "away_avg_goals_scored",
        "away_avg_goals_conceded",
        "away_avg_goal_diff",
        "away_avg_days_since_last_match",
        "prob_h",
        "prob_d",
        "prob_a",
    ]

    # Combine core numerical features with the dummy variable column names
    base_feature_cols = (
        core_numerical_features
        + home_dummies.columns.tolist()
        + away_dummies.columns.tolist()
    )

    # Ensure base_feature_cols contains only unique names
    base_feature_cols = list(set(base_feature_cols))

    # Filter base_feature_cols to only include columns actually present in df
    # (e.g., if prob_h was missing entirely due to no odds data)
    base_feature_cols = [col for col in base_feature_cols if col in df.columns]

    # Convert all feature columns to numeric and fill any remaining NaNs
    for col in base_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            0
        )  # 'coerce' turns non-numeric to NaN
        # fillna(0) for safe GLM training

    # Drop rows that still have NaNs in any of the base_feature_cols (after numeric conversion and fillna)
    df.dropna(subset=base_feature_cols, inplace=True)

    if df.empty:
        print(
            "❌ No complete feature data available after processing; aborting score model training."
        )
        return None

    # --- Train/Test Split (Time-series aware) ---
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # --- Home Goals Model ---
    print("\n🤖 Training Home Goals Poisson Model...")
    X_train_home = (
        df_train[base_feature_cols].astype(float).to_numpy()
    )  # Select all base_feature_cols and convert to float numpy array
    X_train_home = sm.add_constant(X_train_home, prepend=False)  # Add intercept
    poisson_model_home = sm.GLM(
        endog=df_train["fthg"].to_numpy(),  # Also convert endog to numpy
        exog=X_train_home,
        family=sm.families.Poisson(),
    ).fit()

    # --- Away Goals Model ---
    print("🤖 Training Away Goals Poisson Model...")
    X_train_away = (
        df_train[base_feature_cols].astype(float).to_numpy()
    )  # Select all base_feature_cols and convert to float numpy array
    X_train_away = sm.add_constant(X_train_away, prepend=False)  # Add intercept
    poisson_model_away = sm.GLM(
        endog=df_train["ftag"].to_numpy(),  # Also convert endog to numpy
        exog=X_train_away,
        family=sm.families.Poisson(),
    ).fit()

    # --- Evaluation ---
    print("\n📋 Evaluating score models on the test set...")

    # Prepare test data for prediction
    X_test_home = df_test[base_feature_cols].astype(float).to_numpy()
    X_test_home = sm.add_constant(X_test_home, prepend=False)
    X_test_away = df_test[base_feature_cols].astype(float).to_numpy()
    X_test_away = sm.add_constant(X_test_away, prepend=False)

    # Predict mean goals
    home_mean_goals_preds = poisson_model_home.predict(X_test_home)
    away_mean_goals_preds = poisson_model_away.predict(X_test_away)

    # Calculate RMSE on actual goals
    home_rmse = np.sqrt(
        mean_squared_error(df_test["fthg"].to_numpy(), home_mean_goals_preds)
    )
    away_rmse = np.sqrt(
        mean_squared_error(df_test["ftag"].to_numpy(), away_mean_goals_preds)
    )
    print(f"✓ Home Goals RMSE: {home_rmse:.3f}")
    print(f"✓ Away Goals RMSE: {away_rmse:.3f}")

    # Save models
    os.makedirs(SCORE_MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = os.path.join(
        SCORE_MODEL_DIR, SCORE_MODEL_NAME_TEMPLATE.format(timestamp=timestamp)
    )

    score_model_data = {
        "home_model": poisson_model_home,
        "away_model": poisson_model_away,
        "trained_on": timestamp,
        "home_rmse": home_rmse,
        "away_rmse": away_rmse,
        "feature_cols": base_feature_cols,  # Store features used for score models
        "all_teams": all_teams,  # Store categories for prediction
        "home_exog_names": poisson_model_home.model.exog_names,  # Store exog_names
        "away_exog_names": poisson_model_away.model.exog_names,  # Store exog_names
    }

    joblib.dump(score_model_data, path)
    print(f"💾 Score models saved to {path}")

    return path


def load_latest_score_model():
    """Load the most recent trained score model."""
    if not os.path.exists(SCORE_MODEL_DIR):
        raise FileNotFoundError(
            f"Score model directory {SCORE_MODEL_DIR} not found. Train first."
        )

    files = sorted([f for f in os.listdir(SCORE_MODEL_DIR) if f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError("No score model files found. Train first.")

    model_path = os.path.join(SCORE_MODEL_DIR, files[-1])
    score_model_data = joblib.load(model_path)
    print(f"📂 Loaded score models from {model_path}")

    return score_model_data


def predict_score(score_model_data, match_features_df):
    """
    Predicts the most likely score for a single match using the Poisson models,
    given the match's features in a DataFrame.
    Returns the predicted score as a string (e.g., "2-1").
    """
    home_model = score_model_data["home_model"]
    away_model = score_model_data["away_model"]
    all_teams = score_model_data["all_teams"]
    base_feature_cols = score_model_data["feature_cols"]

    # Ensure match_features_df is a single row DataFrame for the match
    predict_data = match_features_df.copy()

    # Prepare a DataFrame with all possible feature columns and fill with 0
    # This ensures consistency with training data, even if a feature isn't in predict_data
    prediction_exog_template = pd.DataFrame(
        0.0, index=predict_data.index, columns=base_feature_cols
    )

    # Populate numerical features
    for col in set(base_feature_cols) & set(predict_data.columns):
        prediction_exog_template[col] = predict_data[col]

    # Create dummy variables for home_team and away_team for the single match
    home_team_cat = pd.Categorical(predict_data["home_team"], categories=all_teams)
    away_team_cat = pd.Categorical(predict_data["away_team"], categories=all_teams)

    home_team_dummies = pd.get_dummies(home_team_cat, prefix="home_team", dtype=int)
    away_team_dummies = pd.get_dummies(away_team_cat, prefix="away_team", dtype=int)

    # Populate dummy features into the template DataFrame
    for col in home_team_dummies.columns:
        if col in prediction_exog_template.columns:
            prediction_exog_template[col] = home_team_dummies[col]
    for col in away_team_dummies.columns:
        if col in prediction_exog_template.columns:
            prediction_exog_template[col] = away_team_dummies[col]

    # Ensure all feature columns are numeric (float) and fill any remaining NaNs (should be 0)
    for col in base_feature_cols:
        prediction_exog_template[col] = pd.to_numeric(
            prediction_exog_template[col], errors="coerce"
        ).fillna(0)

    # Prepare exog for home model prediction
    # Ensure the order of columns matches the model's exog_names
    X_predict_home = (
        prediction_exog_template[home_model.exog_names[1:]].astype(float).to_numpy()
    )  # Exclude constant
    X_predict_home = sm.add_constant(X_predict_home, prepend=False)

    # Prepare exog for away model prediction
    # Ensure the order of columns matches the model's exog_names
    X_predict_away = (
        prediction_exog_template[away_model.exog_names[1:]].astype(float).to_numpy()
    )  # Exclude constant
    X_predict_away = sm.add_constant(X_predict_away, prepend=False)

    try:
        # Predict the expected number of home goals (lambda_home)
        lambda_home = home_model.predict(X_predict_home)[0]

        # Predict the expected number of away goals (lambda_away)
        lambda_away = away_model.predict(X_predict_away)[0]
    except Exception as e:
        if "Score Prediction not available for one or both teams/features" not in str(
            e
        ):  # Avoid duplicate error message
            return (
                f"Score Prediction not available for one or both teams/features ({e})."
            )
        else:
            return (
                f"Score Prediction not available for one or both teams/features ({e})."
            )

    # 3. Calculate the probability matrix and find the most likely score
    max_goals = 5  # Max goals to check for
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):  # Home goals
        for j in range(max_goals + 1):  # Away goals
            # P(Home=i, Away=j) = P(Home=i) * P(Away=j)
            prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(
                j, lambda_away
            )

    # Find the indices of the maximum probability
    predicted_home_goals, predicted_away_goals = np.unravel_index(
        prob_matrix.argmax(), prob_matrix.shape
    )

    return f"{predicted_home_goals}-{predicted_away_goals}"


if __name__ == "__main__":
    train_score_model()
