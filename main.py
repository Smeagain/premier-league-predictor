import argparse
from model import train_model, predict
from config import DEFAULT_PREDICT_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Premier League Predictor CLI")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming matches")
    parser.add_argument("--limit", type=int, default=DEFAULT_PREDICT_LIMIT,
                        help="Number of fixtures to predict")
    parser.add_argument("--json", action="store_true", help="Output predictions as JSON")
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.predict:
        predict(limit=args.limit, as_json=args.json)


if __name__ == "__main__":
    main()
