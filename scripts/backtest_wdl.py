import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss
from features import build_features

TRAIN_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
TEST_SEASON = "2025-2026"

def main():
    df = pd.read_parquet("data/master_dataset.parquet")

    # Keep only normal season labels
    df = df[df["season"].astype(str).str.contains("-")].copy()

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test_df = df[df["season"].eq(TEST_SEASON)].copy()

    train_feat = build_features(df=train_df, drop_na_labels=True)
    test_feat = build_features(df=test_df, drop_na_labels=True)

    feature_cols = [
        c for c in train_feat.columns
        if c not in ["date", "home_team", "away_team", "label", "ftr"]
    ]

    X_train, y_train = train_feat[feature_cols], train_feat["label"].astype(int)
    X_test, y_test = test_feat[feature_cols], test_feat["label"].astype(int)

    model = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        objective="multiclass",
        class_weight="balanced",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    pred = proba.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=[0, 1, 2])

    # Majority-class baseline on test
    majority = y_test.mode().iloc[0]
    base_acc = accuracy_score(y_test, [majority] * len(y_test))

    print("Train seasons:", TRAIN_SEASONS)
    print("Test season:", TEST_SEASON)
    print("Test samples:", len(X_test))
    print(f"Accuracy: {acc:.4f}")
    print(f"Majority baseline accuracy: {base_acc:.4f}")
    print(f"Log loss: {ll:.4f}")
    print("Test class distribution:", y_test.value_counts(normalize=True).sort_index().to_dict())

if __name__ == "__main__":
    main()
