from pathlib import Path

import pandas as pd
import pytest

from src.config import MODEL_PATH
from src.predict import predict_one


def customer(**overrides):
    base = {
        "age": 28,
        "gender": "Male",
        "purchase_count": 8,
        "total_spent": 4200.0,
        "avg_order_value": 525.0,
        "days_since_last_purchase": 12,
        "website_visits_30d": 15,
        "cart_adds_30d": 6,
        "discount_usage_rate": 0.25,
        "support_tickets_90d": 0,
        "customer_tenure_days": 420,
        "preferred_channel": "Mobile",
        "region": "North",
    }
    base.update(overrides)
    return base


def test_model_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Model chưa được train")
    assert MODEL_PATH.exists()


def test_prediction_schema():
    if not MODEL_PATH.exists():
        pytest.skip("Model chưa được train")
    result = predict_one(MODEL_PATH, customer())
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["label"], str)


def test_two_different_customer_cases():
    if not MODEL_PATH.exists():
        pytest.skip("Model chưa được train")
    high_activity = customer()
    low_activity = customer(
        purchase_count=1,
        total_spent=120,
        avg_order_value=120,
        days_since_last_purchase=240,
        website_visits_30d=1,
        cart_adds_30d=0,
        support_tickets_90d=3,
        customer_tenure_days=90,
        preferred_channel="Store",
        region="Central",
    )
    r1 = predict_one(MODEL_PATH, high_activity)
    r2 = predict_one(MODEL_PATH, low_activity)
    assert r1["prediction"] in (0, 1)
    assert r2["prediction"] in (0, 1)
    assert r1["probability"] != r2["probability"]
