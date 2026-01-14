import os

import pandas as pd

def test_core_columns_exist():
    df = pd.read_csv("Season 2023-2024/E0.csv")

    required = {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    }

    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

def test_goals_non_negative():
    df = pd.read_csv("Season 2023-2024/E0.csv")

    for col in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        assert (df[col] >= 0).all(), f"Negative values found in {col}"

def test_core_columns_exist_across_seasons():
    paths = [
        "Season 2023-2024/E0.csv",
        "Season 2025-2026/E0.csv",
    ]

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}

    for path in paths:
        assert os.path.exists(path), f"Missing file: {path}"
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        assert not missing, f"{path} missing required columns: {missing}"
