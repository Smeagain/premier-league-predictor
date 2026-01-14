import requests
import time
import os
import json
from config import HEADERS, COMPETITIONS


def fetch_standings(comp_code, season):
    """
    Fetch standings for a given competition code and season.
    Returns a list of table rows or None if forbidden.
    """
    url = (
        f"https://api.football-data.org/v4/competitions/{comp_code}/standings"
        f"?season={season}"
    )
    res = requests.get(url, headers=HEADERS, timeout=15)
    time.sleep(0.5)  # Add a small delay to avoid rate limiting
    if res.status_code == 403:
        print(f"⚠️ 403 forbidden fetching {comp_code} {season}; skipping.")
        return None
    res.raise_for_status()
    data = res.json()
    # "standings" is a list of groups; each has a "table" key
    rows = []
    for group in data.get("standings", []):
        rows.extend(group.get("table", []))

    return rows


def fetch_all_data(seasons):
    """
    Fetch and flatten standings from all configured competitions/seasons.
    Returns a list of dicts, each with added 'competition' and 'season' keys.
    """
    all_rows = []
    for season in seasons:
        for code in COMPETITIONS:
            table = fetch_standings(code, season)
            if not table:
                continue
            for row in table:
                row["competition"] = code
                row["season"] = season
                all_rows.append(row)

    return all_rows


def fetch_upcoming_fixtures(limit=10):
    """
    Fetch scheduled (future) fixtures across all competitions,
    merge & sort by match date, then return up to `limit` matches.
    """
    fixtures = []
    for comp_code in COMPETITIONS:
        url = (
            f"https://api.football-data.org/v4/competitions/{comp_code}/matches"
            f"?status=SCHEDULED"
        )
        res = requests.get(url, headers=HEADERS, timeout=15)
        time.sleep(0.5)  # Add a small delay to avoid rate limiting
        if res.status_code != 200:
            print(f"⚠️ Unable to fetch fixtures for {comp_code}: {res.status_code}")
            continue
        data = res.json()
        for m in data.get("matches", []):
            odds = m.get("odds") or {}
            fixtures.append(
                {
                    "date": m["utcDate"],
                    "competition": comp_code,
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "odds_h": odds.get("homeWin"),
                    "odds_d": odds.get("draw"),
                    "odds_a": odds.get("awayWin"),
                }
            )

    # sort by date ascending
    fixtures.sort(key=lambda x: x["date"])
    return fixtures[:limit]


def fetch_matches(seasons):
    """
    Fetch historical match data for training.
    Returns a list of match dictionaries with full match information.
    Prioritizes reading from cache to avoid rate limits.
    """
    all_matches = []
    CACHE_DIR = "cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

    for season in seasons:
        for comp_code in COMPETITIONS:
            url_base = (
                f"https://api.football-data.org/v4/competitions/{comp_code}/matches"
                f"?season={season}"
            )
            filename = (
                os.path.join(
                    CACHE_DIR,
                    url_base.replace("https://", "")
                    .replace("/", "")
                    .replace("?", "")
                    .replace("=", ""),
                )
                + ".json"
            )

            data = None
            if os.path.exists(filename):
                try:
                    with open(filename, "r") as f:
                        data = json.load(f)
                    print(f"✓ Loaded from cache: {comp_code} {season}")
                except Exception as e:
                    print(f"⚠️ Error reading cache file {filename}: {e}")

            if data is None:
                # Fallback to API if not in cache or cache read failed
                res = requests.get(url_base, headers=HEADERS, timeout=15)
                time.sleep(0.5)  # Add a small delay to avoid rate limiting

                if res.status_code == 403:
                    print(f"⚠️  403 forbidden fetching {comp_code} {season}; skipping.")
                    continue
                if res.status_code != 200:
                    print(f"⚠️  Error fetching {comp_code} {season}: {res.status_code}")
                    continue

                data = res.json()
                print(f"✓ Fetched from API: {comp_code} {season}")
                # Save to cache
                try:
                    with open(filename, "w") as f:
                        json.dump(data, f)
                except Exception as e:
                    print(f"⚠️ Error writing to cache file {filename}: {e}")

            matches = data.get("matches", [])
            # Only include finished matches with scores
            finished = [
                m
                for m in matches
                if m["status"] == "FINISHED"
                and m["score"]["fullTime"]["home"] is not None
            ]
            all_matches.extend(finished)
            print(f"✓ Total finished matches for {comp_code} {season}: {len(finished)}")

    return all_matches


def load_latest_model():
    """Load the most recent trained model with metadata."""
    import os
    import joblib
    from config import MODEL_DIR

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory {MODEL_DIR} not found. Run training first."
        )

    files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    if not files:
        raise FileNotFoundError("No model files found. Run training first.")

    model_path = os.path.join(MODEL_DIR, files[-1])
    model_data = joblib.load(model_path)
    print(f"📂 Loaded model from {model_path}")

    # Show model info
    if isinstance(model_data, dict):
        print(f"   Trained on: {model_data.get('trained_on', 'unknown')}")
        print(f"   Training samples: {model_data.get('n_samples', 'unknown')}")
    return model_data
