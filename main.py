"""
Customer Segmentation Machine Learning Pipeline.
Main execution entry point providing CLI options and interactive workflow.
"""

import os
import sys
import argparse
from typing import Optional

from src.data_loader import load_and_validate_data
from src.preprocessor import CustomerDataPreprocessor
from src.model import CustomerSegmentationModel
from src.profiler import CustomerSegmentProfiler
from src.visualizer import CustomerSegmentationVisualizer


def run_pipeline(
    data_path: str,
    k: int,
    missing_strategy: str = "impute",
    output_dir: str = "output",
    generate_elbow: bool = False
) -> None:
    """
    Executes the customer segmentation pipeline end-to-end.
    """
    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENTATION MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Dataset path     : {data_path}")
    print(f"Target clusters K: {k}")
    print(f"Missing strategy : {missing_strategy}")
    print(f"Output directory : {output_dir}")
    print("=" * 80 + "\n")

    # Step 1: Ingestion & Validation
    print("[1/5] Loading and validating dataset...")
    df_clean, metadata = load_and_validate_data(data_path, missing_strategy=missing_strategy)
    print(f"  -> Total records loaded: {metadata['initial_rows']}")
    if metadata['missing_records_count'] > 0:
        print(f"  -> Handled {metadata['missing_records_count']} rows with missing values using '{missing_strategy}'.")
        if metadata['imputed_columns']:
            for col, info in metadata['imputed_columns'].items():
                print(f"     * Imputed '{col}': {info['missing_count']} missing cells replaced with median ({info['imputed_value']})")
    print(f"  -> Valid records ready for analysis: {metadata['final_rows']}")

    # Step 2: Preprocessing & Scaling
    print("\n[2/5] Standardizing behavioral features with StandardScaler...")
    preprocessor = CustomerDataPreprocessor()
    scaled_features, feature_names = preprocessor.fit_transform(df_clean)
    print(f"  -> Features scaled: {feature_names}")
    print(f"  -> Feature matrix shape: {scaled_features.shape}")

    # Step 3: K-Means Clustering Model
    print(f"\n[3/5] Training K-Means clustering algorithm (K={k})...")
    model = CustomerSegmentationModel(k=k, random_state=42)
    clustered_df, model_summary = model.fit_predict(scaled_features, df_clean)
    print(f"  -> Model converged with Inertia: {model_summary['inertia']:.2f}")
    print(f"  -> Cluster distribution:")
    for cid, count in sorted(model_summary["cluster_distribution"].items()):
        pct = (count / len(clustered_df)) * 100
        print(f"     * Cluster {cid}: {count} customers ({pct:.1f}%)")

    # Step 4: Business Profiling & Persona Generation
    print("\n[4/5] Computing cluster metrics and business personas...")
    profiler = CustomerSegmentProfiler(clustered_df)
    profile_summary_text = profiler.format_summary_text()
    print(profile_summary_text)

    # Step 5: Export Clustered Data & Visualizations
    print("\n[5/5] Exporting outputs and generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)

    # Export CSV
    csv_out_path = os.path.join(output_dir, "segmented_customers.csv")
    clustered_df.to_csv(csv_out_path, index=False)
    print(f"  -> Exported segmented CSV: {os.path.abspath(csv_out_path)}")

    # Visualization
    visualizer = CustomerSegmentationVisualizer(output_dir=output_dir)

    # Transform centroids back to unscaled space for accurate plotting
    unscaled_centroids = preprocessor.inverse_transform(model.cluster_centers_)
    plot_filename = f"customer_clusters_k{k}.png"
    plot_path = visualizer.plot_clusters(
        clustered_df=clustered_df,
        unscaled_centroids=unscaled_centroids,
        k=k,
        save_filename=plot_filename
    )
    print(f"  -> Saved cluster scatter plot: {plot_path}")

    if generate_elbow:
        elbow_path = visualizer.plot_elbow_curve(
            scaled_features=scaled_features,
            max_k=min(8, len(df_clean)),
            save_filename="elbow_curve.png"
        )
        print(f"  -> Saved elbow curve plot: {elbow_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


def interactive_mode():
    """
    Guides the user through custom inputs interactively.
    """
    print("\n--- Interactive Customer Segmentation Setup ---")
    default_csv = os.path.join("data", "synthetic_customers.csv")
    csv_input = input(f"Enter dataset CSV path [Default: {default_csv}]: ").strip()
    data_path = csv_input if csv_input else default_csv

    while True:
        k_input = input("Enter number of clusters K (e.g., 3): ").strip()
        try:
            k = int(k_input)
            if k <= 0:
                print("Error: K must be a positive integer >= 1.")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer for K.")

    strategy_input = input("Missing value strategy ('impute' or 'drop') [Default: impute]: ").strip().lower()
    strategy = strategy_input if strategy_input in ["impute", "drop"] else "impute"

    elbow_input = input("Generate Elbow Curve plot? (y/n) [Default: y]: ").strip().lower()
    generate_elbow = elbow_input != "n"

    run_pipeline(
        data_path=data_path,
        k=k,
        missing_strategy=strategy,
        output_dir="output",
        generate_elbow=generate_elbow
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised Customer Segmentation Machine Learning Pipeline using K-Means."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "synthetic_customers.csv"),
        help="Path to customer dataset CSV file (default: data/synthetic_customers.csv)"
    )
    parser.add_argument(
        "-k", "--clusters",
        type=int,
        default=3,
        help="Number of clusters K (default: 3)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["impute", "drop"],
        default="impute",
        help="Missing value handling strategy: 'impute' (median) or 'drop' (default: impute)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated CSV and plots (default: output)"
    )
    parser.add_argument(
        "--elbow",
        action="store_true",
        help="Generate and save an Elbow Method curve for optimal K analysis"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive prompts to configure options"
    )

    args = parser.parse_args()

    try:
        if args.interactive or "--interactive" in sys.argv:
            interactive_mode()
        elif len(sys.argv) == 1:
            # If run directly with no arguments, run default pipeline with elbow curve
            run_pipeline(
                data_path=args.data,
                k=args.clusters,
                missing_strategy=args.strategy,
                output_dir=args.output_dir,
                generate_elbow=True
            )
        else:
            run_pipeline(
                data_path=args.data,
                k=args.clusters,
                missing_strategy=args.strategy,
                output_dir=args.output_dir,
                generate_elbow=args.elbow
            )
    except (ValueError, FileNotFoundError) as err:
        print(f"\n[CONFIGURATION ERROR]: {err}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
