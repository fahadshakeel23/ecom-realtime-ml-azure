from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/propensity_model.joblib")

app = FastAPI(title="E-commerce Propensity Scoring API", version="1.0")


class ScoreRequest(BaseModel):
    # Match your feature columns used in training
    events_total: float = 0
    sessions_total: float = 0
    product_views: float = 0
    add_to_carts: float = 0
    checkout_starts: float = 0
    distinct_products_touched: float = 0
    purchases: float = 0

    # Optional metadata (not used by model but useful for logging)
    customer_id: Optional[str] = None
    event_date: Optional[str] = None


class ScoreResponse(BaseModel):
    propensity: float
    model_version: str = "local-dev"
    customer_id: Optional[str] = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train first to generate artifacts/propensity_model.joblib"
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    X = np.array([[
        req.events_total,
        req.sessions_total,
        req.product_views,
        req.add_to_carts,
        req.checkout_starts,
        req.distinct_products_touched,
        req.purchases
    ]])

    proba = float(model.predict_proba(X)[:, 1][0])
    return ScoreResponse(propensity=proba, customer_id=req.customer_id)
