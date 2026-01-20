import json
import random
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

EVENT_TYPES = ["product_view", "search", "add_to_cart", "checkout_start", "purchase"]

PRODUCT_IDS = [f"P{n:04d}" for n in range(1, 200)]
CATEGORIES = ["cookies", "snacks", "health", "fashion", "electronics"]
CHANNELS = ["web", "ios", "android"]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_event(user_id: str, session_id: str) -> Dict[str, Any]:
    event_type = random.choice(EVENT_TYPES)
    base = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "event_ts": utc_now_iso(),
        "user_id": user_id,
        "session_id": session_id,
        "source": random.choice(CHANNELS),
        "device": random.choice(["mobile", "desktop"]),
        "app_version": random.choice(["1.0.0", "1.1.0", "1.2.0"]),
        "ip": "0.0.0.0",
        "geo_country": random.choice(["AE", "US", "IN", "GB"]),
        "properties": {}
    }

    if event_type in ["product_view", "add_to_cart", "purchase"]:
        base["properties"] = {
            "product_id": random.choice(PRODUCT_IDS),
            "category": random.choice(CATEGORIES),
            "price": round(random.uniform(5, 250), 2),
            "currency": "USD"
        }
    elif event_type == "search":
        base["properties"] = {
            "query": random.choice(["protein cookies", "ragi cookies", "hoodie", "skincare", "headphones"]),
            "results_count": random.randint(0, 120)
        }
    elif event_type == "checkout_start":
        base["properties"] = {
            "cart_value": round(random.uniform(10, 500), 2),
            "items": random.randint(1, 7)
        }

    return base

def generate_events(n_users: int = 50, n_events: int = 200) -> List[Dict[str, Any]]:
    events = []
    for _ in range(n_events):
        user_id = f"U{random.randint(1, n_users):05d}"
        session_id = str(uuid.uuid4())
        events.append(make_event(user_id, session_id))
    return events

if __name__ == "__main__":
    events = generate_events()
    for e in events[:5]:
        print(json.dumps(e))
