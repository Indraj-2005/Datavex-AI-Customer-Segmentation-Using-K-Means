"""
Customer Profiling and Business Persona Generation Engine.
Analyzes segmented customer clusters to compute unscaled summary statistics
(mean, median, min, max) and assigns marketing persona labels.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

def assign_persona(
    mean_income: float,
    mean_spending: float,
    median_pop_income: float,
    median_pop_spending: float,
    k: int
) -> str:
    """
    Assigns intuitive marketing persona labels based on cluster relative positioning.
    """
    if k == 1:
        return "Whole Customer Base (Unified Segment)"

    income_high = mean_income >= median_pop_income
    spending_high = mean_spending >= median_pop_spending

    if income_high and spending_high:
        return "High-Value VIPs (High Income, High Spending)"
    elif income_high and not spending_high:
        return "Conservative Savers (High Income, Low Spending)"
    elif not income_high and spending_high:
        return "Carefree Spenders (Low/Moderate Income, High Spending)"
    else:
        return "Budget-Conscious (Low Income, Low Spending)"


class CustomerSegmentProfiler:
    """
    Profiles customer clusters and generates business summaries.
    """

    def __init__(self, clustered_df: pd.DataFrame):
        """
        Parameters:
        -----------
        clustered_df : pd.DataFrame
            Customer dataset with 'cluster_id' assigned.
        """
        if "cluster_id" not in clustered_df.columns:
            raise ValueError("Dataframe must contain 'cluster_id' column for profiling.")
        self.df = clustered_df

    def compute_cluster_profiles(self) -> pd.DataFrame:
        """
        Computes descriptive metrics (count, share, mean, median, min, max)
        for annual income and spending score per cluster.

        Returns:
        --------
        pd.DataFrame:
            Summary table indexed by cluster_id.
        """
        total_customers = len(self.df)
        k = self.df["cluster_id"].nunique()

        overall_median_income = float(self.df["annual_income"].median())
        overall_median_spending = float(self.df["spending_score"].median())

        feature_cols = ["annual_income", "spending_score"]
        if "purchase_frequency" in self.df.columns:
            feature_cols.append("purchase_frequency")

        profiles = []
        for cluster_id in sorted(self.df["cluster_id"].unique()):
            cluster_subset = self.df[self.df["cluster_id"] == cluster_id]
            count = len(cluster_subset)
            percentage = (count / total_customers) * 100

            mean_income = float(cluster_subset["annual_income"].mean())
            mean_spending = float(cluster_subset["spending_score"].mean())

            persona = assign_persona(
                mean_income=mean_income,
                mean_spending=mean_spending,
                median_pop_income=overall_median_income,
                median_pop_spending=overall_median_spending,
                k=k
            )

            profile_entry = {
                "cluster_id": cluster_id,
                "persona": persona,
                "customer_count": count,
                "percentage_share": round(percentage, 2),
                "income_mean": round(mean_income, 2),
                "income_median": round(float(cluster_subset["annual_income"].median()), 2),
                "income_min": round(float(cluster_subset["annual_income"].min()), 2),
                "income_max": round(float(cluster_subset["annual_income"].max()), 2),
                "spending_mean": round(mean_spending, 2),
                "spending_median": round(float(cluster_subset["spending_score"].median()), 2),
                "spending_min": round(float(cluster_subset["spending_score"].min()), 2),
                "spending_max": round(float(cluster_subset["spending_score"].max()), 2),
            }

            if "purchase_frequency" in cluster_subset.columns:
                profile_entry["frequency_mean"] = round(float(cluster_subset["purchase_frequency"].mean()), 2)
                profile_entry["frequency_median"] = round(float(cluster_subset["purchase_frequency"].median()), 2)

            profiles.append(profile_entry)

        profile_df = pd.DataFrame(profiles).set_index("cluster_id")
        return profile_df

    def format_summary_text(self) -> str:
        """
        Formats a clean, human-readable ASCII summary of customer segments.
        """
        profile_df = self.compute_cluster_profiles()
        lines = []
        lines.append("=" * 80)
        lines.append("CUSTOMER SEGMENTATION BUSINESS PROFILE SUMMARY")
        lines.append("=" * 80)

        for cluster_id, row in profile_df.iterrows():
            lines.append(f"\n--- Cluster {cluster_id}: {row['persona']} ---")
            lines.append(f"  * Size: {row['customer_count']} customers ({row['percentage_share']}% of total)")
            lines.append(
                f"  * Annual Income: Mean=${row['income_mean']:,.2f} | "
                f"Median=${row['income_median']:,.2f} | "
                f"Range=[${row['income_min']:,.2f} - ${row['income_max']:,.2f}]"
            )
            lines.append(
                f"  * Spending Score: Mean={row['spending_mean']:.1f} | "
                f"Median={row['spending_median']:.1f} | "
                f"Range=[{row['spending_min']} - {row['spending_max']}]"
            )
            if "frequency_mean" in row:
                lines.append(f"  * Purchase Frequency: Mean={row['frequency_mean']:.1f}/yr")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
