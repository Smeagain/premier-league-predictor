import os
import joblib
import lightgbm as lgb
from datetime import datetime
from config import MODEL_DIR, MODEL_NAME_TEMPLATE
from features import build_features
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score


def train_model():
    """
    Trains a LightGBM model to predict match outcomes using a time-series
    aware cross-validation strategy.
    Target: 0 = Home Win, 1 = Draw, 2 = Away Win
    """
    print("📊 Building feature matrix for training...")
    # 1. Build features using our new pipeline
    df = build_features()

    if df.empty:
        print("❌ Feature matrix is empty; aborting training.")
        return None

    # 2. Define features (X) and target (y)
    # Exclude non-feature columns
    feature_cols = [
        col
        for col in df.columns
        if col not in ["date", "home_team", "away_team", "label", "ftr"]
    ]
    X = df[feature_cols]
    y = df["label"]

    print(f"✓ Using {len(feature_cols)} features for training.")
    print(f"✓ Training data shape: {X.shape}")

    # 3. Set up Time-Series Cross-Validation
    # Use 5 splits; the model will be trained on 4 folds and validated on the 5th.
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"📈 Using TimeSeriesSplit with {n_splits} splits for cross-validation.")

    lgbm_base_params = {
        "random_state": 42,
        "n_jobs": -1,
        "objective": "multiclass",
        "class_weight": "balanced",
    }
    lgbm = lgb.LGBMClassifier(**lgbm_base_params)

    # Define the parameter distributions for RandomizedSearchCV
    param_distributions = {
        "n_estimators": [100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 50],
        "max_depth": [-1, 10, 20],  # -1 means no limit
        "subsample": [0.7, 0.85, 1.0],  # Fraction of data to use for training each tree
        "colsample_bytree": [
            0.7,
            0.85,
            1.0,
        ],  # Fraction of features to use for each tree
    }

    n_iter_search = 15
    print(
        f"\n🚀 Starting RandomizedSearchCV for hyperparameter tuning ({n_iter_search} iterations)..."
    )
    print(f"Parameter distributions: {param_distributions}")

    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=42,  # for reproducibility
    )

    # Fit RandomizedSearchCV on the full dataset
    random_search.fit(X, y)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print(f"\n✅ RandomizedSearchCV completed. Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.3f}")

    # 5. Evaluate on the last time-series split
    # This gives the most realistic estimate of performance on new data.
    print("\n📋 Evaluating best model on the final time-series validation set...")

    # Get the train/test indices for the last split
    train_index, test_index = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # We refit the best model on the training part of the last split
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"✓ Final validation accuracy: {test_acc:.3f}")
    print("\nClassification Report (Final Validation Set):")
    print(
        classification_report(
            y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]
        )
    )

    # 6. Save the final model
    print("\n💾 Saving the trained model...")
    # The best_model is already fitted on the training data of the last split,
    # which is a good representation of a model trained on recent historical data.

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path = os.path.join(MODEL_DIR, MODEL_NAME_TEMPLATE.format(timestamp=timestamp))

    model_data = {
        "model": best_model,  # Save the model already fitted on the last split
        "features": feature_cols,
        "trained_on": timestamp,
        "n_samples": len(X_train),  # n_samples is for the last training set
        "validation_accuracy": test_acc,
        "best_params": best_params,
    }
    joblib.dump(model_data, path)
    print(f"✓ Model saved to {path}")

    return path


if __name__ == "__main__":
    train_model()
