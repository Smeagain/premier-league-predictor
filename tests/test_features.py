import unittest
import pandas as pd
from features import build_features


class TestFeatures(unittest.TestCase):
    def test_away_result_mapping(self):
        """
        Tests that a team's form is correctly calculated from both
        home and away perspectives.
        """
        # Match 1: Team 1 (H) beats Team 2 (A). Team 2 gets 0 points.
        # Match 2: Team 2 (H) draws with Team 3 (A).
        # The test checks Team 2's average points going into the second match.
        data = {
            "date": pd.to_datetime(["2023-01-01T12:00:00Z", "2023-01-08T12:00:00Z"]),
            "home_team": ["Team 1", "Team 2"],
            "away_team": ["Team 2", "Team 3"],
            "fthg": [2, 1],
            "ftag": [1, 1],
            "ftr": ["H", "D"],
        }
        matches_df = pd.DataFrame(data)

        # Call build_features with the DataFrame directly
        features = build_features(df=matches_df, drop_na_labels=False)

        # The second row (index 1) corresponds to the second match where Team 2 is home.
        # Their form should be based on the first match, where they were away and lost (0 pts).
        row = features.iloc[1]

        # Check that the home team is correct
        self.assertEqual(row["home_team"], "Team 2")
        # Check that the average points for the home team (Team 2) is 0, based on the prior loss.
        self.assertEqual(row["home_avg_pts"], 0.0)



if __name__ == "__main__":
    unittest.main()
