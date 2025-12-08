import pandas as pd
from collections import deque
from config import RECENT_FORM_N


def build_features(matches: list) -> pd.DataFrame:
    """
    Given raw match JSON list, compute feature matrix with labels.
    """
    matches = sorted(matches, key=lambda m: m["utcDate"])
    recent = {}
    records = []

    for m in matches:
        home_id = m["homeTeam"]["id"]
        away_id = m["awayTeam"]["id"]
        score = m["score"]["fullTime"]
        label = None
        if score["home"] is not None and score["away"] is not None:
            if score["home"] > score["away"]:
                label = 0
            elif score["home"] == score["away"]:
                label = 1
            else:
                label = 2

        recent.setdefault(home_id, deque(maxlen=RECENT_FORM_N))
        recent.setdefault(away_id, deque(maxlen=RECENT_FORM_N))

        def form_counts(team_id):
            results = list(recent[team_id])
            return results.count(0), results.count(1), results.count(2)

        home_form = form_counts(home_id)
        away_form = form_counts(away_id)

        home_pos = m.get("homeTeam", {}).get("position")
        away_pos = m.get("awayTeam", {}).get("position")

        records.append(
            {
                "home_id": home_id,
                "away_id": away_id,
                "home_form_w": home_form[0],
                "home_form_d": home_form[1],
                "home_form_l": home_form[2],
                "away_form_w": away_form[0],
                "away_form_d": away_form[1],
                "away_form_l": away_form[2],
                "home_pos": home_pos,
                "away_pos": away_pos,
                "label": label,
            }
        )

        if label is not None:
            recent[home_id].append(label)
            # invert away team outcome
            recent[away_id].append(2 - label)

    return pd.DataFrame(records)
