"""
Module for loading and validating customer segmentation datasets.
Handles schema verification, missing values handling (imputation or rejection),
and general data sanity checks according to requirements and test case TC5.
"""

import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["customer_id", "annual_income", "spending_score"]
OPTIONAL_COLUMNS = ["purchase_frequency"]

def load_and_validate_data(
    filepath: str,
    missing_strategy: str = "impute"
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads a customer CSV file, validates required schema, and handles missing values.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
    missing_strategy : str
        Strategy to handle missing values:
        - 'impute': Replaces missing numerical values with column median (robust to outliers).
        - 'drop': Rejects/drops records containing missing values in required columns.

    Returns:
    --------
    Tuple[pd.DataFrame, dict]:
        - Cleaned pandas DataFrame.
        - Summary metadata dictionary reporting row count, missing values handled, etc.

    Raises:
    -------
    FileNotFoundError: If the specified file does not exist.
    ValueError: If required columns are missing or file is empty.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{filepath}': {str(e)}")

    if df.empty:
        raise ValueError(f"The dataset '{filepath}' is completely empty.")

    # Standardize column names (strip whitespace and lowercase)
    column_mapping = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_mapping)

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset missing required column(s): {missing_cols}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    initial_row_count = len(df)
    metadata = {
        "initial_rows": initial_row_count,
        "missing_strategy": missing_strategy,
        "missing_records_count": 0,
        "imputed_columns": {},
        "final_rows": initial_row_count
    }

    # Count missing values across feature columns
    feature_cols = [col for col in REQUIRED_COLUMNS[1:] if col in df.columns]
    if "purchase_frequency" in df.columns:
        feature_cols.append("purchase_frequency")

    total_missing_cells = int(df[feature_cols].isna().sum().sum())
    rows_with_missing = int(df[feature_cols].isna().any(axis=1).sum())
    metadata["missing_records_count"] = rows_with_missing

    if rows_with_missing > 0:
        if missing_strategy == "drop":
            df = df.dropna(subset=feature_cols).copy()
            df = df.reset_index(drop=True)
            metadata["final_rows"] = len(df)
        elif missing_strategy == "impute":
            df = df.copy()
            for col in feature_cols:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    col_median = float(df[col].median())
                    df[col] = df[col].fillna(col_median)
                    metadata["imputed_columns"][col] = {
                        "missing_count": n_missing,
                        "imputed_value": round(col_median, 2)
                    }
            metadata["final_rows"] = len(df)
        else:
            raise ValueError(
                f"Unknown missing_strategy '{missing_strategy}'. Must be 'impute' or 'drop'."
            )

    if len(df) == 0:
        raise ValueError("No valid customer records remain after handling missing values.")

    return df, metadata
