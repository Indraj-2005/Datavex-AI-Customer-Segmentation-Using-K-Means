"""
K-Means Clustering model engine for customer segmentation.
Handles cluster validation (TC3, TC4, TC6), training, centroid computation,
and cluster ID assignments (TC2).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class CustomerSegmentationModel:
    """
    Wrapper for Scikit-Learn KMeans with strict validation and business utility methods.
    """

    def __init__(self, k: int, random_state: int = 42, n_init: int = 10):
        """
        Parameters:
        -----------
        k : int
            Number of customer clusters to generate. Must be an integer >= 1.
        random_state : int
            Seed for random centroid initialization for reproducible clustering.
        n_init : int
            Number of time the k-means algorithm will be run with different centroid seeds.
        """
        self.k = self._validate_k(k)
        self.random_state = random_state
        self.n_init = n_init
        self.model: Optional[KMeans] = None
        self.is_fitted = False

    @staticmethod
    def _validate_k(k: Any) -> int:
        """
        Validates cluster count K according to TC3 and TC4.
        """
        if not isinstance(k, (int, np.integer)):
            # If float passed, check if it's an exact integer representation
            if isinstance(k, float) and k.is_integer():
                k = int(k)
            else:
                raise ValueError(f"Cluster count K must be an integer, got: {type(k).__name__} ({k})")

        if k <= 0:
            raise ValueError(
                f"Invalid cluster count K={k}. The number of clusters must be a positive integer >= 1."
            )
        return int(k)

    def fit_predict(
        self,
        scaled_features: np.ndarray,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trains the K-Means model and assigns cluster labels to the dataframe.

        Parameters:
        -----------
        scaled_features : np.ndarray
            Scaled feature matrix from preprocessor of shape (n_samples, n_features).
        df : pd.DataFrame
            Original unscaled customer dataset to append 'cluster_id' to.

        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame with added 'cluster_id' column.
            - Metrics dictionary (inertia, cluster counts, silhouette score if k > 1).
        """
        n_samples = scaled_features.shape[0]

        # TC6 Validation: Records count N < K
        if n_samples < self.k:
            raise ValueError(
                f"Number of clusters (k={self.k}) cannot be greater than the number of "
                f"customer records (n_samples={n_samples}). Provide at least {self.k} records."
            )

        # Initialize and fit KMeans (TC2, TC3)
        self.model = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init=self.n_init
        )
        labels = self.model.fit_predict(scaled_features)
        self.is_fitted = True

        df_clustered = df.copy()
        df_clustered["cluster_id"] = labels

        # Cluster distribution counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

        summary = {
            "k": self.k,
            "n_samples": n_samples,
            "inertia": float(self.model.inertia_),
            "cluster_distribution": cluster_distribution,
            "n_iter": int(self.model.n_iter_)
        }

        return df_clustered, summary

    @property
    def cluster_centers_(self) -> np.ndarray:
        """
        Returns scaled cluster centroids from the fitted KMeans model.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.cluster_centers_

    @property
    def inertia_(self) -> float:
        """
        Returns model inertia (sum of squared distances to closest centroid).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return float(self.model.inertia_)
