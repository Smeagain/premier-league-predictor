import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API settings
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
COMPETITION_PL = "PL"
COMPETITION_CHAMP = "ELC"

# Seasons configuration
PL_SEASONS = [2023, 2024]
CHAMP_SEASONS = [2022, 2023]

# Data settings
RECENT_FORM_N = 5  # number of past matches to consider

# Model settings
MODEL_DIR = "models"
MODEL_NAME_TEMPLATE = "pl_model_{date}.pkl"

# CLI defaults
DEFAULT_PREDICT_LIMIT = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
