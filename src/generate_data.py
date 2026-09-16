from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_PATH, RANDOM_STATE


def generate_dataset(n_samples: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 66, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.48, 0.52])
    purchase_count = rng.poisson(6, n_samples) + 1
    total_spent = np.round(rng.gamma(shape=2.5, scale=450, size=n_samples), 2)
    avg_order_value = np.round(total_spent / purchase_count, 2)
    days_since_last_purchase = np.clip(rng.gamma(2.0, 30, n_samples).astype(int), 1, 365)
    website_visits_30d = rng.poisson(8, n_samples)
    cart_adds_30d = rng.poisson(3, n_samples)
    discount_usage_rate = np.round(rng.beta(2, 5, n_samples), 3)
    support_tickets_90d = rng.poisson(0.7, n_samples)
    customer_tenure_days = rng.integers(30, 1500, n_samples)
    preferred_channel = rng.choice(["Web", "Mobile", "Store"], n_samples, p=[0.35, 0.45, 0.20])
    region = rng.choice(["North", "Central", "South"], n_samples, p=[0.34, 0.20, 0.46])

    # Synthetic behavioral relationship for educational purposes.
    score = (
        0.45 * purchase_count
        + 0.00025 * total_spent
        - 0.045 * days_since_last_purchase
        + 0.08 * website_visits_30d
        + 0.12 * cart_adds_30d
        - 0.75 * discount_usage_rate
        - 0.35 * support_tickets_90d
        + 0.0003 * customer_tenure_days
        + rng.normal(0, 1.7, n_samples)
        - 2.2
    )
    probability = 1 / (1 + np.exp(-score))
    returned_90d = rng.binomial(1, probability)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n_samples + 1)],
        "age": age,
        "gender": gender,
        "purchase_count": purchase_count,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "days_since_last_purchase": days_since_last_purchase,
        "website_visits_30d": website_visits_30d,
        "cart_adds_30d": cart_adds_30d,
        "discount_usage_rate": discount_usage_rate,
        "support_tickets_90d": support_tickets_90d,
        "customer_tenure_days": customer_tenure_days,
        "preferred_channel": preferred_channel,
        "region": region,
        "returned_90d": returned_90d,
    })

    # Simulate a small amount of missing data so the imputer is actually exercised.
    for col in ["website_visits_30d", "discount_usage_rate", "preferred_channel"]:
        mask = rng.random(n_samples) < 0.015
        df.loc[mask, col] = np.nan

    # Simulate a few duplicates to demonstrate data cleaning.
    duplicates = df.sample(n=10, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {DATA_PATH}")
    print(df.head())
    print("Target distribution:")
    print(df["returned_90d"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
