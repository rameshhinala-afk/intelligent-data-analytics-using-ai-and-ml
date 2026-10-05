"""Generate a demo customer dataset (with missing values, duplicates and outliers)."""
import numpy as np
import pandas as pd


def make_sample(n=800, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    tenure = rng.integers(1, 72, n)
    monthly = np.round(rng.normal(65, 20, n).clip(15, 150), 2)
    support_calls = rng.poisson(2, n)
    contract = rng.choice(["Monthly", "Yearly", "Two-year"], n, p=[0.55, 0.3, 0.15])
    city = rng.choice(["Hyderabad", "Mumbai", "Delhi", "Bengaluru", "Chennai"], n)
    logit = (-1 + 0.025 * monthly - 0.04 * tenure + 0.35 * support_calls
             + np.where(contract == "Monthly", 1.0, -0.5))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"age": age, "tenure_months": tenure, "monthly_charges": monthly,
                       "support_calls": support_calls, "contract": contract, "city": city,
                       "total_spend": np.round(monthly * tenure + rng.normal(0, 50, n), 2),
                       "churn": np.where(churn == 1, "Yes", "No")})
    df.loc[rng.choice(n, 30, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "city"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "total_spend"] *= 8          # outliers
    return pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # duplicates


if __name__ == "__main__":
    make_sample().to_csv("sample_customers.csv", index=False)
    print("Saved sample_customers.csv")
