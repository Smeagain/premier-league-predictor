import pandas as pd

# K-factor determines how much Elo ratings change after a match.
# A higher K means more volatility.
K_FACTOR = 20

# The divisor in the Elo expectation formula. 400 is standard for chess and widely used.
ELO_DIVISOR = 400

# The default starting Elo for any new team.
DEFAULT_ELO = 1500


def get_expected_score(team_elo, opponent_elo):
    """
    Calculates the expected score (win probability) for a team based on Elo ratings.
    """
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / ELO_DIVISOR))


def update_elo(team_elo, opponent_elo, actual_score):
    """
    Updates a team's Elo rating after a match.
    """
    expected_score = get_expected_score(team_elo, opponent_elo)
    return team_elo + K_FACTOR * (actual_score - expected_score)


def calculate_elo_ratings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates historical Elo ratings for teams based on match results.

    Args:
        matches_df: DataFrame containing match data, sorted by date.
                    Must include 'home_team', 'away_team', and 'ftr' columns.

    Returns:
        DataFrame with two new columns: 'home_elo' and 'away_elo'.
    """
    print("Calculating Elo ratings...")

    # Ensure dataframe is sorted by date
    matches_df = matches_df.sort_values("date").reset_index(drop=True)

    elo_ratings = {}
    home_elos = []
    away_elos = []

    # Iterate over each match to calculate and update Elo ratings
    for index, row in matches_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Get current Elo ratings, defaulting if a team is new
        home_elo = elo_ratings.get(home_team, DEFAULT_ELO)
        away_elo = elo_ratings.get(away_team, DEFAULT_ELO)

        # Store the Elo ratings *before* this match
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        # Determine the actual score from the result
        if row["ftr"] == "H":
            home_actual_score = 1.0
            away_actual_score = 0.0
        elif row["ftr"] == "A":
            home_actual_score = 0.0
            away_actual_score = 1.0
        else:  # Draw
            home_actual_score = 0.5
            away_actual_score = 0.5

        # Calculate new Elo ratings
        new_home_elo = update_elo(home_elo, away_elo, home_actual_score)
        new_away_elo = update_elo(away_elo, home_elo, away_actual_score)

        # Update the master Elo dictionary
        elo_ratings[home_team] = new_home_elo
        elo_ratings[away_team] = new_away_elo

    matches_df["home_elo"] = home_elos
    matches_df["away_elo"] = away_elos

    print("✓ Elo rating calculation complete. Added 'home_elo' and 'away_elo' columns.")
    return matches_df


if __name__ == "__main__":
    # Example usage:
    # This demonstrates how to use the function. It reads the master dataset,
    # calculates Elo ratings, and prints the last 5 rows with the new columns.
    try:
        master_df = pd.read_parquet("data/master_dataset.parquet")
        elo_df = calculate_elo_ratings(master_df)
        print("\nElo ratings for the last 5 matches:")
        print(elo_df[["date", "home_team", "away_team", "home_elo", "away_elo"]].tail())
    except FileNotFoundError:
        print("Error: 'data/master_dataset.parquet' not found.")
        print("Please run data_integration.py first to create the master dataset.")
