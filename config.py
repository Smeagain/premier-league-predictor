from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing FOOTBALL_API_KEY in environment or .env file")

HEADERS = {"X-Auth-Token": API_KEY}

# Define all competitions to include
COMPETITIONS = {
    "PL": "Premier League",
    "ELC": "Championship",
}

MODEL_DIR = "models"
MODEL_NAME_TEMPLATE = "model_{timestamp}.pkl"

# number of recent matches to consider for form features
RECENT_FORM_N = 5
