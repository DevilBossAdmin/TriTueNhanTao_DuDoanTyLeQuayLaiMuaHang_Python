from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH

FEATURES = [
    "age", "gender", "purchase_count", "total_spent", "avg_order_value",
    "days_since_last_purchase", "website_visits_30d", "cart_adds_30d",
    "discount_usage_rate", "support_tickets_90d", "customer_tenure_days",
    "preferred_channel", "region",
]


def predict_one(model_path: Path, customer: dict) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model chưa tồn tại: {model_path}. Hãy chạy `python -m src.generate_data` và `python -m src.train` trước."
        )
    model = joblib.load(model_path)
    X = pd.DataFrame([customer], columns=FEATURES)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Có khả năng quay lại" if prediction else "Khả năng quay lại thấp",
        "probability": probability,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Customer return prediction")
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--gender", choices=["Male", "Female"], required=True)
    p.add_argument("--purchase-count", type=int, required=True)
    p.add_argument("--total-spent", type=float, required=True)
    p.add_argument("--avg-order-value", type=float, required=True)
    p.add_argument("--days-since-last-purchase", type=int, required=True)
    p.add_argument("--website-visits-30d", type=int, required=True)
    p.add_argument("--cart-adds-30d", type=int, required=True)
    p.add_argument("--discount-usage-rate", type=float, required=True)
    p.add_argument("--support-tickets-90d", type=int, required=True)
    p.add_argument("--customer-tenure-days", type=int, required=True)
    p.add_argument("--preferred-channel", choices=["Web", "Mobile", "Store"], required=True)
    p.add_argument("--region", choices=["North", "Central", "South"], required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    customer = {
        "age": args.age,
        "gender": args.gender,
        "purchase_count": args.purchase_count,
        "total_spent": args.total_spent,
        "avg_order_value": args.avg_order_value,
        "days_since_last_purchase": args.days_since_last_purchase,
        "website_visits_30d": args.website_visits_30d,
        "cart_adds_30d": args.cart_adds_30d,
        "discount_usage_rate": args.discount_usage_rate,
        "support_tickets_90d": args.support_tickets_90d,
        "customer_tenure_days": args.customer_tenure_days,
        "preferred_channel": args.preferred_channel,
        "region": args.region,
    }
    result = predict_one(MODEL_PATH, customer)
    print(f"Prediction: {result['label']}")
    print(f"Probability of return: {result['probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
