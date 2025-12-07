from model import train_model, predict
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.predict:
        predict(limit=args.limit)


if __name__ == '__main__':
    main()
