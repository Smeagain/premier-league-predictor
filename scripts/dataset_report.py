import sys
import glob
import pandas as pd

league = sys.argv[1] if len(sys.argv) > 1 else "E0"
paths = sorted(glob.glob(f"Season */{league}.csv"))

assert paths, f"No files found for league {league}"

for path in paths:
    df = pd.read_csv(path)

    print("=" * 60)
    print("File:", path)
    print("Rows, Columns:", df.shape)

    dates = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    print("Date min:", dates.min())
    print("Date max:", dates.max())
    print("Unparseable dates:", dates.isna().sum())

    key_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    print("Missing rate (key cols):")
    print(df[key_cols].isna().mean())
