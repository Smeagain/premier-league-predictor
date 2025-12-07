import requests
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
                row['competition'] = code
                row['season'] = season
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
        if res.status_code != 200:
            print(f"⚠️ Unable to fetch fixtures for {comp_code}: HTTP {res.status_code}")
            continue
        data = res.json()
        for m in data.get("matches", []):
            fixtures.append({
                "date": m["utcDate"],
                "competition": comp_code,
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"]
            })

    # sort by date ascending
    fixtures.sort(key=lambda x: x["date"])
    return fixtures[:limit]

