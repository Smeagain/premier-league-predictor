import pandas as pd
from config import RECENT_FORM_N
from elo import calculate_elo_ratings

# Known Premier League teams by season are inferred by match count:
# PL has 380 matches per season, Championship has 552


def infer_is_championship(df: pd.DataFrame) -> pd.Series:
    # Championship seasons have >380 matches
    season_counts = df.groupby("season")["home_team"].transform("count")
    return season_counts > 380


def build_features(
    data_path="data/master_dataset.parquet", df=None, drop_na_labels=True
) -> pd.DataFrame:
    """
    Builds a feature matrix from the master dataset or a provided DataFrame.

    Args:
        data_path (str): The path to the master dataset parquet file.
                         Used if `df` is None.
        df (pd.DataFrame): An optional DataFrame to use directly. If provided,
                           `data_path` is ignored.
        drop_na_labels (bool): If True, rows with NaN labels (e.g., future matches)
                                will be dropped. Set to False for prediction.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a match,
                      with features for home and away teams and a label.
    """
    if df is None:
        if data_path is None:
            raise ValueError("Either 'data_path' or 'df' must be provided.")
        try:
            df = pd.read_parquet(data_path)
        except FileNotFoundError:
            print(f"Error: Master dataset not found at {data_path}")
            print("Please run data_integration.py first.")
            return pd.DataFrame()
    else:
        # If df is provided, ensure it's a copy to avoid modifying original
        df = df.copy()

    if "season" in df.columns:
        df["is_championship"] = infer_is_championship(df).astype(int)
    else:
        df["is_championship"] = 0

    # 1. Calculate Elo ratings
    df = calculate_elo_ratings(df)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # --- Prepare for rolling calculations ---
    # Create a unified list of matches from the perspective of each team
    # This makes it easier to calculate rolling stats for each team individually
    df_home = df.copy()
    df_away = df.copy()
    df_home["team"] = df_home["home_team"]
    df_home["opponent"] = df_home["away_team"]
    df_home["is_home"] = 1
    df_home["goals_scored"] = df_home["fthg"]
    df_home["goals_conceded"] = df_home["ftag"]

    df_away["team"] = df_away["away_team"]
    df_away["opponent"] = df_away["home_team"]
    df_away["is_home"] = 0
    df_away["goals_scored"] = df_away["ftag"]
    df_away["goals_conceded"] = df_away["fthg"]

    # Map 'H', 'D', 'A' to points for the team in perspective
    pts_map = {"H": 3, "D": 1, "A": 0}
    df_home["pts"] = df_home["ftr"].map(pts_map)
    df_away["pts"] = df_away["ftr"].map({k: 3 - v for k, v in pts_map.items()})

    # Combine and sort to process chronologically for each team
    team_perspective_df = pd.concat([df_home, df_away]).sort_values(["team", "date"])

    grouped = team_perspective_df.groupby("team")
    team_perspective_df["days_since_last_match"] = grouped["date"].diff().dt.days
    # Fill NaNs for the first match of each team with a neutral value (e.g., 2 weeks)
    team_perspective_df["days_since_last_match"] = team_perspective_df[
        "days_since_last_match"
    ].fillna(14)

    # --- Calculate Rolling Features ---
    # The shift(1) ensures we use data *before* the current match
    print("Calculating rolling features...")

    rolling_cols = ["pts", "goals_scored", "goals_conceded", "days_since_last_match"]
    for col in rolling_cols:
        team_perspective_df[f"avg_{col}"] = (
            grouped[col]
            .rolling(window=RECENT_FORM_N, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
        )

    team_perspective_df["avg_goal_diff"] = (
        team_perspective_df["avg_goals_scored"]
        - team_perspective_df["avg_goals_conceded"]
    )

    # --- Merge Features back into Match-level DataFrame ---
    # We now have rolling stats for each team for each match.
    # We need to merge them back into the original match-centric format.
    features_df = df[
        ["date", "home_team", "away_team", "ftr", "elo_diff", "is_championship"]
    ].copy()

    # Get home team features
    home_features = team_perspective_df[team_perspective_df["is_home"] == 1].copy()
    home_feature_map = {
        "avg_pts": "home_avg_pts",
        "avg_goals_scored": "home_avg_goals_scored",
        "avg_goals_conceded": "home_avg_goals_conceded",
        "avg_goal_diff": "home_avg_goal_diff",
        "avg_days_since_last_match": "home_avg_days_since_last_match",
    }
    home_features.rename(columns=home_feature_map, inplace=True)

    # Get away team features
    away_features = team_perspective_df[team_perspective_df["is_home"] == 0].copy()
    away_feature_map = {
        "avg_pts": "away_avg_pts",
        "avg_goals_scored": "away_avg_goals_scored",
        "avg_goals_conceded": "away_avg_goals_conceded",
        "avg_goal_diff": "away_avg_goal_diff",
        "avg_days_since_last_match": "away_avg_days_since_last_match",
    }
    away_features.rename(columns=away_feature_map, inplace=True)

    # Merge home and away features
    cols_to_merge = ["date", "home_team", "away_team"]
    features_df = features_df.merge(
        home_features[cols_to_merge + list(home_feature_map.values())],
        on=cols_to_merge,
    )
    features_df = features_df.merge(
        away_features[cols_to_merge + list(away_feature_map.values())],
        on=cols_to_merge,
    )

    # --- Calculate Betting Odds Features ---
    # Calculate implied probability from bookmaker odds
    print("Calculating implied probabilities from betting odds...")
    odds_cols = ["b365h", "b365d", "b365a"]
    if all(col in df.columns for col in odds_cols):
        # Calculate inverse of the odds
        inv_h, inv_d, inv_a = 1 / df["b365h"], 1 / df["b365d"], 1 / df["b365a"]

        # Calculate the margin (overround)
        margin = inv_h + inv_d + inv_a - 1

        # Calculate implied probabilities
        df["prob_h"] = inv_h / (1 + margin)
        df["prob_d"] = inv_d / (1 + margin)
        df["prob_a"] = inv_a / (1 + margin)

        # Add to the features dataframe
        features_df = features_df.merge(
            df[["date", "home_team", "away_team", "prob_h", "prob_d", "prob_a"]],
            on=["date", "home_team", "away_team"],
        )
    else:
        print("⚠️ Betting odds columns not found. Skipping probability features.")

    # --- Final Touches ---
    # Add label for modeling
    label_map = {"H": 0, "D": 1, "A": 2}
    features_df["label"] = df["ftr"].map(label_map)

    # If dropping NaN labels, remove rows where 'label' is NaN
    if drop_na_labels:
        features_df = features_df.dropna(subset=["label"])

    print(
        f"✓ Feature engineering complete. Final feature matrix shape: {
            features_df.shape}")
    return features_df
