import pandas as pd

df = pd.read_parquet("data/master_dataset.parquet", columns=["season"])

# keep only normal season labels like "2023-2024"
seasons = sorted([s for s in df["season"].dropna().unique() if "-" in str(s)])

print("Backtest seasons:", seasons)
