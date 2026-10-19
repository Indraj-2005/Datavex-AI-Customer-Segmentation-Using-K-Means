"""
Feature preprocessing pipeline for customer segmentation.
Handles extraction of numerical features, scaling via StandardScaler,
and inverse transformations for reporting unscaled centroids.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["annual_income", "spending_score"]
ALL_POTENTIAL_FEATURES = ["annual_income", "spending_score", "purchase_frequency"]

class CustomerDataPreprocessor:
    """
    Standardizes numerical customer behavior features to zero-mean and unit-variance.
    Standardization ensures features with larger raw magnitudes (such as annual income in $10,000s)
    do not disproportionately bias Euclidean distance calculations over spending scores (1-100).
    """

    def __init__(self, feature_columns: Optional[List[str]] = None):
        """
        Parameters:
        -----------
        feature_columns : Optional[List[str]]
            List of feature column names to scale and use for clustering.
            If None, defaults to ['annual_income', 'spending_score'].
        """
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.is_fitted = False

    def select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Resolves feature columns based on what is available in the dataframe.
        """
        if self.feature_columns is not None:
            # Verify user-specified features exist
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Requested feature column(s) not found in dataset: {missing}")
            return self.feature_columns

        # Default to core 2D segmentation features (annual_income, spending_score)
        available = [col for col in DEFAULT_FEATURES if col in df.columns]
        if not available:
            raise ValueError(
                f"No valid numeric feature columns found in dataset. "
                f"Expected at least one of: {DEFAULT_FEATURES}"
            )
        return available

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fits the scaler on the selected feature columns and transforms the data.

        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned customer dataset.

        Returns:
        --------
        Tuple[np.ndarray, List[str]]:
            - Scaled feature matrix of shape (n_samples, n_features).
            - List of feature column names utilized.
        """
        features_to_use = self.select_feature_columns(df)
        self.feature_columns = features_to_use

        raw_features = df[features_to_use].to_numpy(dtype=float)
        scaled_features = self.scaler.fit_transform(raw_features)
        self.is_fitted = True

        return scaled_features, features_to_use

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms features using the already fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform first.")
        if not self.feature_columns:
            raise RuntimeError("No feature columns configured.")

        raw_features = df[self.feature_columns].to_numpy(dtype=float)
        return self.scaler.transform(raw_features)

    def inverse_transform(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Inverse-transforms scaled coordinates (e.g. cluster centroids) back
        to their original, unscaled business domain values.

        Parameters:
        -----------
        scaled_features : np.ndarray
            Scaled feature matrix or centroid array.

        Returns:
        --------
        np.ndarray:
            Original scale coordinates.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform first.")
        return self.scaler.inverse_transform(scaled_features)
