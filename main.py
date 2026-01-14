from predictor import predict_upcoming_matches as predict
from model import train_model
from score_model import train_score_model
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.train:
        train_model()
        train_score_model()
    if args.predict:
        results = predict(limit=args.limit)
        if results:
            print("\n--- Match Predictions ---")
            for r in results:
                prob_str = f"H: {r['probabilities']['home_win']:.2f} | D: {r['probabilities']['draw']:.2f} | A: {r['probabilities']['away_win']:.2f}"
                print(f"📅 {r['date']} - {r['competition']}")
                print(f"   {r['home']} vs {r['away']}")
                print(f"   Prediction: {r['prediction']} ({prob_str})")
                print(f"   Score: {r['score_prediction']}")
            print("-------------------------")


if __name__ == "__main__":
    main()
