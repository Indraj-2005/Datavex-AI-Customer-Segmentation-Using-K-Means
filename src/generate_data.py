"""
Script to generate synthetic customer datasets for K-Means Customer Segmentation.
Generates:
1. Main realistic synthetic customer dataset (data/synthetic_customers.csv)
2. TC1 test dataset: clearly separated low, medium, and high spending groups
3. TC5 test dataset: customer records with missing annual_income values
4. TC6 test dataset: dataset with fewer records than requested clusters
"""

import os
import numpy as np
import pandas as pd

def generate_main_dataset(filepath: str, n_samples: int = 200, random_state: int = 42):
    """
    Generates a realistic multi-segment customer dataset with 5 natural clusters:
    - Low income, Low spending (Sensible/Frugal)
    - Low income, High spending (Carefree)
    - Medium income, Medium spending (Standard)
    - High income, Low spending (Savers/Careful)
    - High income, High spending (VIP/High-Value)
    """
    np.random.seed(random_state)
    n_per_cluster = n_samples // 5

    # 1. Low Income, Low Spending
    inc_1 = np.random.normal(loc=25, scale=5, size=n_per_cluster)
    sp_1 = np.random.normal(loc=20, scale=6, size=n_per_cluster)
    freq_1 = np.random.poisson(lam=3, size=n_per_cluster)

    # 2. Low Income, High Spending
    inc_2 = np.random.normal(loc=26, scale=5, size=n_per_cluster)
    sp_2 = np.random.normal(loc=80, scale=7, size=n_per_cluster)
    freq_2 = np.random.poisson(lam=8, size=n_per_cluster)

    # 3. Middle Income, Middle Spending
    inc_3 = np.random.normal(loc=55, scale=8, size=n_per_cluster)
    sp_3 = np.random.normal(loc=50, scale=7, size=n_per_cluster)
    freq_3 = np.random.poisson(lam=5, size=n_per_cluster)

    # 4. High Income, Low Spending
    inc_4 = np.random.normal(loc=90, scale=10, size=n_per_cluster)
    sp_4 = np.random.normal(loc=20, scale=6, size=n_per_cluster)
    freq_4 = np.random.poisson(lam=4, size=n_per_cluster)

    # 5. High Income, High Spending
    inc_5 = np.random.normal(loc=92, scale=10, size=n_per_cluster)
    sp_5 = np.random.normal(loc=85, scale=6, size=n_per_cluster)
    freq_5 = np.random.poisson(lam=12, size=n_per_cluster)

    incomes = np.concatenate([inc_1, inc_2, inc_3, inc_4, inc_5]) * 1000  # in dollars
    spendings = np.concatenate([sp_1, sp_2, sp_3, sp_4, sp_5])
    frequencies = np.concatenate([freq_1, freq_2, freq_3, freq_4, freq_5])

    # Bound spending score between 1 and 100, frequencies >= 1
    spendings = np.clip(np.round(spendings), 1, 100).astype(int)
    frequencies = np.clip(frequencies, 1, 30).astype(int)
    incomes = np.clip(np.round(incomes, -2), 15000, 150000).astype(int)

    customer_ids = [f"CUST_{i+1:04d}" for i in range(len(incomes))]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "annual_income": incomes,
        "spending_score": spendings,
        "purchase_frequency": frequencies
    })

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated main synthetic dataset: {filepath} ({len(df)} rows)")
    return df

def generate_test_case_datasets(base_dir: str):
    """Generates specific test case datasets for TC1, TC5, and TC6."""
    os.makedirs(base_dir, exist_ok=True)

    # TC1: Dataset with clearly separated low-, medium-, and high-spending groups
    np.random.seed(101)
    tc1_data = {
        "customer_id": [f"TC1_{i+1:03d}" for i in range(60)],
        "annual_income": (
            [35000 + np.random.randint(-2000, 2000) for _ in range(20)] +   # Low tier
            [60000 + np.random.randint(-2000, 2000) for _ in range(20)] +   # Medium tier
            [95000 + np.random.randint(-2000, 2000) for _ in range(20)]     # High tier
        ),
        "spending_score": (
            [np.random.randint(5, 25) for _ in range(20)] +     # Low
            [np.random.randint(40, 60) for _ in range(20)] +    # Medium
            [np.random.randint(80, 100) for _ in range(20)]     # High
        ),
        "purchase_frequency": [np.random.randint(1, 15) for _ in range(60)]
    }
    df_tc1 = pd.DataFrame(tc1_data)
    tc1_path = os.path.join(base_dir, "tc1_separated_spending.csv")
    df_tc1.to_csv(tc1_path, index=False)
    print(f"Generated TC1 dataset: {tc1_path} ({len(df_tc1)} rows)")

    # TC5: Missing income for some customers
    np.random.seed(105)
    df_tc5 = df_tc1.copy()
    df_tc5.loc[5, "annual_income"] = np.nan
    df_tc5.loc[15, "annual_income"] = np.nan
    tc5_path = os.path.join(base_dir, "tc5_missing_income.csv")
    df_tc5.to_csv(tc5_path, index=False)
    print(f"Generated TC5 dataset: {tc5_path} ({len(df_tc5)} rows with 2 missing incomes)")

    # TC6: Dataset containing fewer valid records than requested clusters (e.g. 2 records)
    df_tc6 = pd.DataFrame({
        "customer_id": ["TC6_001", "TC6_002"],
        "annual_income": [45000, 75000],
        "spending_score": [30, 85],
        "purchase_frequency": [3, 9]
    })
    tc6_path = os.path.join(base_dir, "tc6_few_records.csv")
    df_tc6.to_csv(tc6_path, index=False)
    print(f"Generated TC6 dataset: {tc6_path} ({len(df_tc6)} rows)")

if __name__ == "__main__":
    generate_main_dataset("data/synthetic_customers.csv")
    generate_test_case_datasets("data/test_cases")
