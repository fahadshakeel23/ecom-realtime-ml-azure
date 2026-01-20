import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import load_yaml


def build_dataset(features_csv: str, labels_csv: str) -> pd.DataFrame:
    feats = pd.read_csv(features_csv)
    labs = pd.read_csv(labels_csv)

    # join on customer_id + event_date
    df = feats.merge(
        labs[["customer_id", "event_date", "label_purchase_next_7d"]],
        on=["customer_id", "event_date"],
        how="inner",
    )
    return df


def split_data(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, cfg: dict):
    """
    Supports:
      - time split (preferred when you have multiple dates)
      - user split (fallback when you only have 1 date)
    """
    split_cfg = cfg.get("split", {})
    method = split_cfg.get("method", "time")  # default to time, but we'll fallback if impossible

    # ---- Time split ----
    if method == "time":
        # If config doesn't have dates or dataset has only one date, fallback to user split
        if "train_end_date" not in cfg or "test_start_date" not in cfg:
            method = "user"
        else:
            # If only one unique date in dataset, time split will fail -> fallback
            unique_dates = df["event_date"].dt.date.nunique()
            if unique_dates < 2:
                method = "user"

    if method == "time":
        train_end = pd.to_datetime(cfg["train_end_date"])
        test_start = pd.to_datetime(cfg["test_start_date"])

        train_mask = df["event_date"] <= train_end
        test_mask = df["event_date"] >= test_start

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        return X_train, y_train, X_test, y_test, {"split_method": "time"}

    # ---- User split (recommended fallback) ----
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))

    rng = np.random.default_rng(random_state)
    users = df["customer_id"].dropna().unique()

    if len(users) < 2:
        raise ValueError("Not enough unique customers to do user-based split. Need at least 2 users.")

    rng.shuffle(users)

    n_test = max(1, int(len(users) * test_size))
    test_users = set(users[:n_test])

    test_mask = df["customer_id"].isin(test_users)
    train_mask = ~test_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    return X_train, y_train, X_test, y_test, {"split_method": "user", "test_size": test_size, "random_state": random_state}


def main():
    cfg = load_yaml("configs/train_dev.yaml")

    # Local CSV exports
    features_csv = "artifacts/features_customer_daily.csv"
    labels_csv = "artifacts/labels_purchase_7d.csv"

    df = build_dataset(features_csv, labels_csv)

    # Basic cleaning
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "customer_id"])  # keep rows with valid join keys

    target_col = cfg.get("target_col", "label_purchase_next_7d")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    y = df[target_col].astype(int)

    # Select feature columns (numeric only) - keep it simple for baseline
    drop_cols = ["customer_id", "event_date", target_col, "last_event_ts_that_day"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure all feature cols are numeric (convert non-numeric to NaN, fill with 0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    # Split
    X_train, y_train, X_test, y_test, split_info = split_data(df, X, y, cfg)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/test split produced empty set. Check split configuration or dataset size.")

    # Model
    model_cfg = cfg.get("model", {})
    max_iter = int(model_cfg.get("max_iter", 500))

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=max_iter))
    ])

    model.fit(X_train, y_train)

    # Metrics
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        **split_info,
        "feature_count": int(X.shape[1])
    }

    # Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/propensity_model.joblib")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
