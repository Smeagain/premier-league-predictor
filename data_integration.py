import os
import glob
import pandas as pd
from api import fetch_matches


def create_master_dataset():
    """
    Loads historical data from legacy CSVs and new data from the API,
    transforms them to a unified schema, combines them, and saves
    to a master Parquet file.
    """
    print("Starting data integration process...")

    # 1. Load and process legacy CSVs
    historical_df = load_historical_csvs()
    print(f"✓ Loaded {len(historical_df)} records from historical CSVs.")

    # 2. Load and process data from the API
    # We'll fetch data for seasons that are likely in our CSVs to test the overlap
    api_df = load_api_data(seasons=[2022, 2023])
    print(f"✓ Loaded {len(api_df)} records from the API.")

    # 3. Combine and save
    master_df = (
        pd.concat([historical_df, api_df])
        .drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="last",  # Keep the API version in case of overlap
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    os.makedirs("data", exist_ok=True)
    master_df.to_parquet("data/master_dataset.parquet")

    print(f"✓ Data integration complete. Master dataset has {len(master_df)} records.")
    print("✓ Saved master dataset to data/master_dataset.parquet")


def load_api_data(seasons):
    """
    Fetches matches for the given seasons from the API and transforms
    the data into a standardized DataFrame.
    """
    matches = fetch_matches(seasons=seasons)

    processed_matches = []
    for m in matches:
        # Derive full-time result
        fthg = m["score"]["fullTime"]["home"]
        ftag = m["score"]["fullTime"]["away"]
        if fthg > ftag:
            ftr = "H"
        elif fthg < ftag:
            ftr = "A"
        else:
            ftr = "D"

        odds = m.get("odds") or {}
        processed_matches.append(
            {
                "date": pd.to_datetime(m["utcDate"]),
                "season": str(m["season"]["startDate"][:4]),
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "fthg": fthg,
                "ftag": ftag,
                "ftr": ftr,
                "hthg": m["score"]["halfTime"]["home"],
                "htag": m["score"]["halfTime"]["away"],
                "b365h": odds.get("homeWin"),
                "b365d": odds.get("draw"),
                "b365a": odds.get("awayWin"),
            }
        )

    return pd.DataFrame(processed_matches)


def load_historical_csvs():
    """
    Finds all E0.csv and E1.csv files, loads them, and transforms
    them into a standardized DataFrame.
    """
    csv_files = glob.glob("Season*/**/*.csv", recursive=True)

    historical_dfs = []
    for f in csv_files:
        season = f.split("/")[0].replace("Season ", "")

        df = pd.read_csv(f, on_bad_lines="skip")

        # Rename columns to unified schema
        df = df.rename(
            columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "fthg",
                "FTAG": "ftag",
                "FTR": "ftr",
                "HTHG": "hthg",
                "HTAG": "htag",
                "B365H": "b365h",
                "B365D": "b365d",
                "B365A": "b365a",
            }
        )

        # Add season and ensure date is in datetime format
        df["season"] = season
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.tz_localize("UTC")

        # Select a consistent set of columns
        # We will add more stats later if they are consistently available
        core_cols = [
            "date",
            "season",
            "home_team",
            "away_team",
            "fthg",
            "ftag",
            "ftr",
            "hthg",
            "htag",
            "b365h",
            "b365d",
            "b365a",
        ]

        # Filter for columns that actually exist in the dataframe
        existing_cols = [col for col in core_cols if col in df.columns]
        df = df[existing_cols]

        historical_dfs.append(df)

    if not historical_dfs:
        return pd.DataFrame()

    return pd.concat(historical_dfs).reset_index(drop=True)


if __name__ == "__main__":
    create_master_dataset()
