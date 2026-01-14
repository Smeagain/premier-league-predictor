import sys
import glob
import pandas as pd

league = sys.argv[1] if len(sys.argv) > 1 else "E0"
paths = sorted(glob.glob(f"Season */{league}.csv"))
assert paths, f"No files found for league {league}"

for path in paths:
    df = pd.read_csv(path)
    rates = df["FTR"].value_counts(normalize=True)

    home = rates.get("H", 0.0)
    draw = rates.get("D", 0.0)
    away = rates.get("A", 0.0)

    print("=" * 60)
    print("File:", path)
    print("Matches:", len(df))
    print(f"Home win: {home:.4f}  Draw: {draw:.4f}  Away win: {away:.4f}")
