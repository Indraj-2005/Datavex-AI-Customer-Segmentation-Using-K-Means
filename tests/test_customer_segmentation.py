"""
Unit and Integration Test Suite for Customer Segmentation Pipeline.
Verifies compliance with test cases TC1 through TC6 and edge conditions.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_and_validate_data
from src.preprocessor import CustomerDataPreprocessor
from src.model import CustomerSegmentationModel
from src.profiler import CustomerSegmentProfiler
from src.visualizer import CustomerSegmentationVisualizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_CASES_DIR = os.path.join(DATA_DIR, "test_cases")

TC1_PATH = os.path.join(TEST_CASES_DIR, "tc1_separated_spending.csv")
TC5_PATH = os.path.join(TEST_CASES_DIR, "tc5_missing_income.csv")
TC6_PATH = os.path.join(TEST_CASES_DIR, "tc6_few_records.csv")
SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_customers.csv")


# ==============================================================================
# Mandatory Test Cases TC1 to TC6
# ==============================================================================

def test_tc1_separated_spending_groups():
    """
    TC1: Dataset with clearly separated low, medium, high-spending groups.
    Expected: Clusters cleanly reflect these separated groups without mixing.
    """
    df, meta = load_and_validate_data(TC1_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled_features, _ = preprocessor.fit_transform(df)

    model = CustomerSegmentationModel(k=3, random_state=42)
    clustered_df, summary = model.fit_predict(scaled_features, df)

    # Check that each spending tier is mapped to a distinct cluster
    low_spenders = clustered_df[clustered_df["spending_score"] < 35]
    med_spenders = clustered_df[(clustered_df["spending_score"] >= 35) & (clustered_df["spending_score"] <= 65)]
    high_spenders = clustered_df[clustered_df["spending_score"] > 65]

    low_cluster = low_spenders["cluster_id"].unique()
    med_cluster = med_spenders["cluster_id"].unique()
    high_cluster = high_spenders["cluster_id"].unique()

    # Each tier should have exactly one assigned cluster ID, and all 3 IDs must be distinct
    assert len(low_cluster) == 1, f"Low spenders split across clusters: {low_cluster}"
    assert len(med_cluster) == 1, f"Medium spenders split across clusters: {med_cluster}"
    assert len(high_cluster) == 1, f"High spenders split across clusters: {high_cluster}"

    assigned_clusters = {low_cluster[0], med_cluster[0], high_cluster[0]}
    assert assigned_clusters == {0, 1, 2}, f"Unexpected clusters assigned: {assigned_clusters}"


def test_tc2_k_equals_3():
    """
    TC2: User inputs K = 3.
    Expected: Every valid customer is mapped to one of the 3 cluster IDs {0, 1, 2}.
    """
    df, meta = load_and_validate_data(SYNTHETIC_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled_features, _ = preprocessor.fit_transform(df)

    model = CustomerSegmentationModel(k=3, random_state=42)
    clustered_df, summary = model.fit_predict(scaled_features, df)

    assigned_labels = set(clustered_df["cluster_id"].unique())
    assert assigned_labels == {0, 1, 2}, f"Expected {0, 1, 2}, got {assigned_labels}"
    assert len(clustered_df) == len(df), "Row count changed unexpectedly"
    assert clustered_df["cluster_id"].isna().sum() == 0, "Found unassigned (NaN) clusters"


def test_tc3_k_equals_1():
    """
    TC3: User inputs K = 1.
    Expected: Produces a single unified cluster (all labeled 0) without crashing.
    """
    df, meta = load_and_validate_data(SYNTHETIC_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled_features, _ = preprocessor.fit_transform(df)

    model = CustomerSegmentationModel(k=1, random_state=42)
    clustered_df, summary = model.fit_predict(scaled_features, df)

    assert clustered_df["cluster_id"].nunique() == 1
    assert set(clustered_df["cluster_id"].unique()) == {0}
    assert summary["cluster_distribution"] == {0: len(df)}


def test_tc4_invalid_k_zero_or_negative():
    """
    TC4: User inputs K <= 0 or invalid non-integer values.
    Expected: Rejects input immediately with a clean validation ValueError.
    """
    with pytest.raises(ValueError, match="positive integer"):
        CustomerSegmentationModel(k=0)

    with pytest.raises(ValueError, match="positive integer"):
        CustomerSegmentationModel(k=-5)

    with pytest.raises(ValueError, match="must be an integer"):
        CustomerSegmentationModel(k="three")

    with pytest.raises(ValueError, match="must be an integer"):
        CustomerSegmentationModel(k=2.5)


def test_tc5_missing_income_handling():
    """
    TC5: Dataset with missing values (e.g. null income).
    Expected: Gracefully handled via documented strategy (impute with median or drop).
    """
    # 1. Imputation Strategy
    df_imputed, meta_impute = load_and_validate_data(TC5_PATH, missing_strategy="impute")
    assert meta_impute["missing_records_count"] == 2
    assert "annual_income" in meta_impute["imputed_columns"]
    assert df_imputed["annual_income"].isna().sum() == 0
    assert len(df_imputed) == meta_impute["initial_rows"]

    # Verify clustering completes smoothly on imputed data
    preprocessor = CustomerDataPreprocessor()
    scaled, _ = preprocessor.fit_transform(df_imputed)
    model = CustomerSegmentationModel(k=3, random_state=42)
    clustered_df, summary = model.fit_predict(scaled, df_imputed)
    assert len(clustered_df) == len(df_imputed)

    # 2. Drop Strategy
    df_dropped, meta_drop = load_and_validate_data(TC5_PATH, missing_strategy="drop")
    assert len(df_dropped) == meta_drop["initial_rows"] - 2
    assert df_dropped["annual_income"].isna().sum() == 0


def test_tc6_few_records_n_less_than_k():
    """
    TC6: Dataset where number of records N < K (e.g., 2 records, K = 3).
    Expected: Gracefully caught with an informative ValueError.
    """
    df, meta = load_and_validate_data(TC6_PATH)
    assert len(df) == 2

    preprocessor = CustomerDataPreprocessor()
    scaled, _ = preprocessor.fit_transform(df)

    model = CustomerSegmentationModel(k=3)
    with pytest.raises(ValueError, match="cannot be greater than the number of customer records"):
        model.fit_predict(scaled, df)


# ==============================================================================
# Edge Cases & Auxiliary Component Tests
# ==============================================================================

def test_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("non_existent_file.csv")


def test_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("id,spending\n1,50\n2,60\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_and_validate_data(str(bad_csv))


def test_empty_file_raises_valueerror(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("customer_id,annual_income,spending_score\n")
    with pytest.raises(ValueError, match="empty"):
        load_and_validate_data(str(empty_csv))


def test_preprocessor_scaling_and_inverse_transform():
    df, _ = load_and_validate_data(SYNTHETIC_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled, cols = preprocessor.fit_transform(df)

    # Scaled features should have near-zero mean and unit variance
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-7)

    # Inverse transform should reconstruct original features
    original_reconstructed = preprocessor.inverse_transform(scaled)
    assert np.allclose(original_reconstructed, df[cols].to_numpy(dtype=float))


def test_profiler_and_business_personas():
    df, _ = load_and_validate_data(SYNTHETIC_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled, _ = preprocessor.fit_transform(df)
    model = CustomerSegmentationModel(k=3, random_state=42)
    clustered_df, _ = model.fit_predict(scaled, df)

    profiler = CustomerSegmentProfiler(clustered_df)
    profile_df = profiler.compute_cluster_profiles()

    assert len(profile_df) == 3
    assert "persona" in profile_df.columns
    assert "percentage_share" in profile_df.columns
    assert sum(profile_df["customer_count"]) == len(df)
    assert abs(sum(profile_df["percentage_share"]) - 100.0) < 0.1

    summary_text = profiler.format_summary_text()
    assert "CUSTOMER SEGMENTATION BUSINESS PROFILE SUMMARY" in summary_text


def test_visualizer_generates_image_files(tmp_path):
    df, _ = load_and_validate_data(SYNTHETIC_PATH)
    preprocessor = CustomerDataPreprocessor()
    scaled, _ = preprocessor.fit_transform(df)
    model = CustomerSegmentationModel(k=3, random_state=42)
    clustered_df, _ = model.fit_predict(scaled, df)

    out_dir = str(tmp_path / "viz_out")
    viz = CustomerSegmentationVisualizer(output_dir=out_dir)

    cluster_plot_path = viz.plot_clusters(clustered_df, k=3, save_filename="test_clusters.png")
    assert os.path.isfile(cluster_plot_path)
    assert os.path.getsize(cluster_plot_path) > 1000

    elbow_plot_path = viz.plot_elbow_curve(scaled, max_k=5, save_filename="test_elbow.png")
    assert os.path.isfile(elbow_plot_path)
    assert os.path.getsize(elbow_plot_path) > 1000
