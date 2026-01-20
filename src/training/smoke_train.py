import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Tiny synthetic dataset (fast + deterministic)
    rng = np.random.default_rng(42)
    n = 200

    X = pd.DataFrame({
        "events_total": rng.integers(1, 50, size=n),
        "sessions_total": rng.integers(1, 10, size=n),
        "product_views": rng.integers(0, 30, size=n),
        "add_to_carts": rng.integers(0, 10, size=n),
        "checkout_starts": rng.integers(0, 5, size=n),
        "distinct_products_touched": rng.integers(0, 20, size=n),
        "purchases": rng.integers(0, 3, size=n),
    })

    # Synthetic label: higher carts/checkouts => more likely purchase
    score = 0.4*X["add_to_carts"] + 0.7*X["checkout_starts"] + 0.2*X["product_views"]
    prob = 1 / (1 + np.exp(-(score - score.mean())/score.std()))
    y = (rng.random(n) < prob).astype(int)

    # Split
    idx = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=300))
    ])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/smoke_model.joblib")
    with open("artifacts/smoke_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Smoke training complete:", metrics)

if __name__ == "__main__":
    main()
