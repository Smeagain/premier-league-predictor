import pandas as pd

paths = [
    "Season 2023-2024/E0.csv",
    "Season 2025-2026/E0.csv",
]

results = {}

for path in paths:
    df = pd.read_csv(path)
    rates = df["FTR"].value_counts(normalize=True)
    results[path] = {
        "Home": rates.get("H", 0.0),
        "Draw": rates.get("D", 0.0),
        "Away": rates.get("A", 0.0),
    }

summary = pd.DataFrame(results).T
summary["Home_shift"] = summary["Home"] - summary.iloc[0]["Home"]
summary["Draw_shift"] = summary["Draw"] - summary.iloc[0]["Draw"]
summary["Away_shift"] = summary["Away"] - summary.iloc[0]["Away"]

print(summary)
