import argparse
import sys
from typing import Optional

from model import train_model
from predictor import predict_upcoming_matches as predict


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="plp", description="Premier League Predictor CLI"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train the model")

    p_predict = sub.add_parser("predict", help="List upcoming fixtures (uses API)")
    p_predict.add_argument(
        "--limit", type=int, default=10, help="Number of fixtures to fetch"
    )

    p_pm = sub.add_parser(
        "predict-match", help="Predict single match by team ids (if supported)"
    )
    p_pm.add_argument("--home", type=int, required=True, help="Home team id")
    p_pm.add_argument("--away", type=int, required=True, help="Away team id")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        print("Training model...")
        path = train_model()
        if path:
            print(f"Model saved to: {path}")
        return 0

    if args.cmd == "predict":
        print(
            f"Fetching next {
                args.limit} fixtures and predicting (if model supports it)..."
        )
        results = predict(limit=args.limit)
        if results:
            print("\n--- Match Predictions ---")
            for r in results:
                prob_str = f"H: {
                    r['probabilities']['home_win']:.2f} | D: {
                    r['probabilities']['draw']:.2f} | A: {
                    r['probabilities']['away_win']:.2f}"
                print(f"📅 {r['date']} - {r['competition']}")
                print(f"   {r['home']} vs {r['away']}")
                print(f"   Prediction: {r['prediction']} ({prob_str})")
                print(f"   Score: {r['score_prediction']}")
            print("-------------------------")
        return 0

    if args.cmd == "predict-match":
        # Try to call a match-level predict if available
        try:
            res = predict(args.home, args.away)  # some versions accept (home, away)
            if res is None:
                print("No prediction available with current model/version.")
            else:
                print("Prediction:")
                for k, v in res.items():
                    print(f"  {k}: {v:.4f}")
        except TypeError:
            print("This model version does not support match-level prediction.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
