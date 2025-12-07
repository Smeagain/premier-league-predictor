import unittest
from features import build_features

class TestFeatures(unittest.TestCase):
    def test_away_result_mapping(self):
        # First match: Team 1 home beats Team 2 -> Team 1 win (0), Team 2 loss (2)
        # Second match: Team 2 home vs Team 3 -> Team 2's prior form should show 1 loss
        matches = [
            {"utcDate": "2023-01-01T12:00:00Z", "homeTeam": {"id": 1}, "awayTeam": {"id": 2}, "score": {"fullTime": {"home": 2, "away": 1}}},
            {"utcDate": "2023-01-08T12:00:00Z", "homeTeam": {"id": 2}, "awayTeam": {"id": 3}, "score": {"fullTime": {"home": 1, "away": 1}}},
        ]

        df = build_features(matches)
        # The second row corresponds to the second match
        row = df.iloc[1]
        self.assertEqual(row['home_id'], 2)
        # Team 2 should have one prior loss recorded (home_form_l == 1)
        self.assertEqual(row['home_form_w'], 0)
        self.assertEqual(row['home_form_d'], 0)
        self.assertEqual(row['home_form_l'], 1)

if __name__ == '__main__':
    unittest.main()
