# Premier League Predictor

Python machine-learning project for football (soccer) match prediction:

- **Outcome** prediction: Home / Draw / Away
- **Scoreline** prediction: Poisson-style goal models (lambdas)

Key modules:
- `features.py` (feature engineering)
- `elo.py` (Elo ratings / diff)
- `model.py` (W/D/L model)
- `score_model.py` (Poisson score model)
- `predictor.py` (prediction pipeline)

Data layout:
- Season CSVs in `Season YYYY-YYYY/` (e.g. `Season 2023-2024/E0.csv`)
- Consolidated dataset expected at `data/master_dataset.parquet` (not committed)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
