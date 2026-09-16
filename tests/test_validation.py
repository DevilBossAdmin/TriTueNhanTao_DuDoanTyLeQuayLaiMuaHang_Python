import pytest

from src.validation import validate_customer


def customer(**overrides):
    payload = {
        "age": 28, "gender": "Male", "purchase_count": 8,
        "total_spent": 4200.0, "avg_order_value": 525.0,
        "days_since_last_purchase": 12, "website_visits_30d": 15,
        "cart_adds_30d": 6, "discount_usage_rate": 0.25,
        "support_tickets_90d": 0, "customer_tenure_days": 420,
        "preferred_channel": "Mobile", "region": "North",
    }
    payload.update(overrides)
    return payload


def test_accepts_valid_customer():
    validate_customer(customer())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"discount_usage_rate": 1.1}, "discount_usage_rate"),
     ({"age": 17}, "age"),
     ({"region": "East"}, "region")],
)
def test_rejects_invalid_customer(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_customer(customer(**overrides))
