"""Shared validation constants for the customer-return project."""
from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from .config import TARGET

NUMERIC_FEATURES = [
    "age", "purchase_count", "total_spent", "avg_order_value",
    "days_since_last_purchase", "website_visits_30d", "cart_adds_30d",
    "discount_usage_rate", "support_tickets_90d", "customer_tenure_days",
]
CATEGORICAL_FEATURES = ["gender", "preferred_channel", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALLOWED_CATEGORIES = {
    "gender": {"Male", "Female"},
    "preferred_channel": {"Web", "Mobile", "Store"},
    "region": {"North", "Central", "South"},
}


def validate_training_data(df: pd.DataFrame) -> None:
    required = set(FEATURES + [TARGET, "customer_id"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset không có dòng dữ liệu.")
    if df[TARGET].isna().any() or set(df[TARGET].dropna().unique()).difference({0, 1}):
        raise ValueError(f"{TARGET} chỉ được chứa 0 hoặc 1 và không được thiếu.")


def validate_customer(customer: Mapping[str, object]) -> None:
    missing = [feature for feature in FEATURES if feature not in customer]
    extra = sorted(set(customer).difference(FEATURES))
    if missing or extra:
        messages = []
        if missing:
            messages.append(f"thiếu: {', '.join(missing)}")
        if extra:
            messages.append(f"không hợp lệ: {', '.join(extra)}")
        raise ValueError("Trường đầu vào " + "; ".join(messages) + ".")
    for feature in NUMERIC_FEATURES:
        value = customer[feature]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{feature} phải là một số hữu hạn.")
        if float(value) < 0:
            raise ValueError(f"{feature} không được âm.")
    if not 18 <= float(customer["age"]) <= 100:
        raise ValueError("age phải nằm trong khoảng 18-100.")
    if not 0 <= float(customer["discount_usage_rate"]) <= 1:
        raise ValueError("discount_usage_rate phải nằm trong khoảng 0-1.")
    for feature, allowed in ALLOWED_CATEGORIES.items():
        if customer[feature] not in allowed:
            raise ValueError(f"{feature} phải là một trong: {', '.join(sorted(allowed))}.")
