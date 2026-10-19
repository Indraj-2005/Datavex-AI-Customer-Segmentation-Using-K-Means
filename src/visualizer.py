"""
Visualization engine for customer segmentation.
Generates 2D cluster scatter plots with overlaid unscaled centroids,
and computes Elbow Curves (Inertia vs. K) for optimal cluster selection.
"""

import os
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Set style
sns.set_theme(style="whitegrid")

class CustomerSegmentationVisualizer:
    """
    Produces publication-quality visualizations of clustered customer data.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_clusters(
        self,
        clustered_df: pd.DataFrame,
        unscaled_centroids: Optional[np.ndarray] = None,
        k: Optional[int] = None,
        save_filename: str = "customer_clusters.png",
        show: bool = False
    ) -> str:
        """
        Generates 2D scatter plot of annual_income vs spending_score colored by cluster_id.
        Overlays unscaled centroids.

        Parameters:
        -----------
        clustered_df : pd.DataFrame
            Dataset containing 'annual_income', 'spending_score', and 'cluster_id'.
        unscaled_centroids : Optional[np.ndarray]
            Array of shape (k, 2) containing unscaled centroid coordinates.
            If None, centroids will be calculated from cluster means.
        k : Optional[int]
            Cluster count for title.
        save_filename : str
            Filename within self.output_dir.
        show : bool
            Whether to display the plot interactively.

        Returns:
        --------
        str:
            Absolute filepath of the saved plot.
        """
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("Dataframe must contain 'cluster_id' to plot clusters.")

        if k is None:
            k = clustered_df["cluster_id"].nunique()

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        # Discrete color palette
        palette = sns.color_palette("tab10", n_colors=max(k, 1))

        # Scatter plot for customer data points
        sns.scatterplot(
            data=clustered_df,
            x="annual_income",
            y="spending_score",
            hue="cluster_id",
            palette=palette,
            s=80,
            alpha=0.85,
            edgecolor="w",
            linewidth=0.8,
            ax=ax,
            legend="full"
        )

        # Compute or use unscaled centroids
        if unscaled_centroids is None:
            centroids_list = []
            for cid in sorted(clustered_df["cluster_id"].unique()):
                sub = clustered_df[clustered_df["cluster_id"] == cid]
                centroids_list.append([sub["annual_income"].mean(), sub["spending_score"].mean()])
            centroids = np.array(centroids_list)
        else:
            centroids = unscaled_centroids

        # Overlay centroids with distinct large star markers
        for cid, center in enumerate(centroids):
            ax.scatter(
                center[0],
                center[1],
                s=300,
                c="gold",
                marker="*",
                edgecolors="black",
                linewidth=1.8,
                zorder=10,
                label="Centroid" if cid == 0 else ""
            )
            ax.annotate(
                f"C{cid}",
                (center[0], center[1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6, edgecolor="none")
            )

        ax.set_title(
            f"Customer Segmentation Clusters (K = {k})",
            fontsize=15,
            fontweight="bold",
            pad=15
        )
        ax.set_xlabel("Annual Income ($)", fontsize=12, labelpad=10)
        ax.set_ylabel("Spending Score (1 - 100)", fontsize=12, labelpad=10)

        # Currency formatting for x axis
        ax.xaxis.set_major_formatter("${x:,.0f}")

        # Legend styling
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=[f"Cluster {lbl}" if lbl.isdigit() else lbl for lbl in labels],
            title="Segments",
            title_fontsize="11",
            loc="upper right",
            frameon=True,
            framealpha=0.9
        )

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        fig.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

        return os.path.abspath(save_path)

    def plot_elbow_curve(
        self,
        scaled_features: np.ndarray,
        max_k: int = 8,
        save_filename: str = "elbow_curve.png"
    ) -> str:
        """
        Computes and plots the Elbow Method (Inertia vs K) to determine optimal K.

        Parameters:
        -----------
        scaled_features : np.ndarray
            Scaled feature matrix.
        max_k : int
            Maximum number of clusters to test.
        save_filename : str
            Destination filename.

        Returns:
        --------
        str:
            Absolute filepath of the saved plot.
        """
        n_samples = len(scaled_features)
        max_k = min(max_k, n_samples)
        k_values = list(range(1, max_k + 1))
        inertias = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_features)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(k_values, inertias, marker="o", linewidth=2.2, color="#1f77b4", markersize=7)
        ax.set_title("Elbow Method For Optimal K", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Clusters (K)", fontsize=11)
        ax.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=11)
        ax.set_xticks(k_values)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return os.path.abspath(save_path)
