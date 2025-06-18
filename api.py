import requests
from config import FOOTBALL_API_KEY, COMPETITION_PL, COMPETITION_CHAMP, PL_SEASONS, CHAMP_SEASONS

HEADERS = {"X-Auth-Token": FOOTBALL_API_KEY}


def fetch_matches(competition: str, season: int, status: str = "FINISHED"):
    """
    Fetch matches for a given competition and season with the specified status.
    """
    url = f"https://api.football-data.org/v4/competitions/{competition}/matches?season={season}&status={status}"
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()
    return res.json().get("matches", [])


def fetch_standings(competition: str, season: int):
    """
    Fetch final standings for a competition season.
    """
    url = f"https://api.football-data.org/v4/competitions/{competition}/standings?season={season}"
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()
    data = res.json()
    return data.get("standings", [])[0].get("table", [])


def get_promoted_teams():
    """
    Identify teams promoted from Championship to PL for configured seasons.
    Returns a set of team IDs.
    """
    promoted = set()
    for year in CHAMP_SEASONS:
        table = fetch_standings(COMPETITION_CHAMP, year)
        for row in table[:3]:
            promoted.add(row["team"]["id"])
    return promoted


def fetch_all_data():
    """
    Fetch and return raw match data for PL and relevant Championship matches.
    """
    pl_matches = []
    for year in PL_SEASONS:
        pl_matches.extend(fetch_matches(COMPETITION_PL, year, status="FINISHED"))

    promoted = get_promoted_teams()
    champ_matches = []
    for year in CHAMP_SEASONS:
        matches = fetch_matches(COMPETITION_CHAMP, year, status="FINISHED")
        champ_matches.extend([m for m in matches if m["homeTeam"]["id"] in promoted or m["awayTeam"]["id"] in promoted])

    return pl_matches + champ_matches
