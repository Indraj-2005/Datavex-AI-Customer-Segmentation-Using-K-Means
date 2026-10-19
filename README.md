# Datavex AI - Customer Segmentation Using K-Means

Unsupervised Machine Learning pipeline to segment customers based on purchasing behavior and annual earnings to support targeted business marketing strategies.

---

## 1. Project Overview

This project implements an end-to-end, unsupervised customer segmentation system using **K-Means Clustering**. By analyzing customer behavioral patterns—principally **Annual Income** and **Spending Score** (with optional **Purchase Frequency**)—the pipeline groups customers into actionable business personas (e.g., *High-Value VIPs*, *Carefree Spenders*, *Conservative Savers*, and *Budget-Conscious Customers*).

### Key Features
* **Modular Architecture**: Clean separation between data ingestion, preprocessing, modeling, business profiling, and visualization.
* **Feature Scaling**: Standardizes features using Z-score normalization (`StandardScaler`), ensuring annual income ($15,000–$150,000) does not disproportionately dominate spending score (1–100) in Euclidean distance calculations.
* **Dynamic Cluster Configuration ($K$)**: Accepts any valid integer $K \ge 1$ with input validation.
* **Missing Value Strategy (TC5)**: Documented handling using either robust median imputation or record dropping.
* **Business Profiling & Persona Generation**: Computes unscaled cluster statistics (mean, median, min, max, percentage share) and assigns marketing persona labels.
* **High-Resolution 2D Visualizations**: Automatically generates scatter plots with overlaid unscaled cluster centroids ($\bigstar$) and computes the Elbow Method curve.
* **100% Automated Pytest Coverage**: Comprehensive automated test suite validating all mandatory test cases (**TC1–TC6**) and edge conditions.

---

## 2. Project Structure

```text
detrax gaming/
│
├── Problem Statement - Indraj A Gouda.pdf   # Original challenge specification
├── README.md                                 # Complete documentation, guide & test reports
├── requirements.txt                          # Project dependencies
├── main.py                                   # Main CLI entry point
│
├── data/
│   ├── synthetic_customers.csv               # 200-sample realistic customer dataset
│   └── test_cases/                           # Dedicated datasets for specific test conditions
│       ├── tc1_separated_spending.csv        # TC1: clearly separated low/med/high spending
│       ├── tc5_missing_income.csv            # TC5: dataset containing missing income entries
│       └── tc6_few_records.csv               # TC6: edge case with N < K records
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Ingestion, schema validation & missing data handling
│   ├── preprocessor.py                       # StandardScaler pipeline & inverse transformation
│   ├── model.py                              # K-Means clustering engine & input validation
│   ├── profiler.py                           # Business persona labeling & statistical summarizer
│   ├── visualizer.py                         # 2D cluster scatter plot & Elbow curve generator
│   └── generate_data.py                      # Synthetic dataset generator
│
├── tests/
│   ├── __init__.py
│   └── test_customer_segmentation.py         # Pytest suite validating TC1 through TC6
│
└── output/                                   # Generated outputs
    ├── customer_clusters_k3.png              # 2D cluster scatter plot with centroids
    ├── elbow_curve.png                       # Elbow Method plot for optimal K evaluation
    └── segmented_customers.csv               # Exported dataset with assigned cluster_id
```

---

## 3. Technology Stack

* **Language**: Python 3.10+ (Tested on Python 3.14.2)
* **Data Processing**: `pandas`, `numpy`
* **Machine Learning**: `scikit-learn` (`KMeans`, `StandardScaler`)
* **Visualization**: `matplotlib`, `seaborn`
* **Automated Testing**: `pytest`

---

## 4. Setup & Installation

### Step 1: Clone or Open Workspace
Ensure you are in the project root directory:
```powershell
cd "detrax gaming"
```

### Step 2: Install Dependencies
Install all required libraries via `pip`:
```powershell
pip install -r requirements.txt
```

---

## 5. Execution Guide

### 5.1 Run Default End-to-End Pipeline
Executes segmentation on the synthetic customer dataset with $K=3$, generating both cluster plots and the elbow curve:
```powershell
python main.py
```

### 5.2 Dynamic Cluster Count ($K$)
Specify custom cluster counts using `--clusters` or `-k`:
```powershell
python main.py -k 4
python main.py -k 5
```

### 5.3 Custom Dataset & Missing Value Strategy
Pass your own CSV file and choose your missing value strategy (`impute` or `drop`):
```powershell
python main.py --data data/test_cases/tc5_missing_income.csv -k 3 --strategy impute
```

### 5.4 Interactive CLI Mode
If you prefer a step-by-step guided setup in the terminal:
```powershell
python main.py --interactive
```

### 5.5 Optional Elbow Method Curve
Generate the Elbow Curve plot to assess optimal $K$ from 1 to 8:
```powershell
python main.py -k 3 --elbow
```

---

## 6. Core Concepts & Implementation Decisions

### 6.1 Why Feature Scaling is Critical
* **Problem**: In customer data, `annual_income` is typically measured in tens of thousands of dollars ($\$15,000 - \$150,000$), while `spending_score` is scaled between $1$ and $100$.
* **Impact**: K-Means relies on Euclidean distance:
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
  Without scaling, differences of $\$1,000$ in income completely swamp a 50-point difference in spending score, rendering the spending score meaningless.
* **Solution**: `CustomerDataPreprocessor` applies `StandardScaler` (Z-score normalization: $z = \frac{x - \mu}{\sigma}$) so both features contribute equally to cluster formation. When reporting centroids and summary metrics, unscaled coordinates are preserved.

### 6.2 Missing Data Handling Strategy (TC5)
Customer records occasionally have missing values (e.g., unrecorded income). `src/data_loader.py` implements two documented strategies:
1. **Median Imputation (`missing_strategy='impute'`, Default)**:
   * Replaces missing numerical values with the feature column's median.
   * Median is chosen over mean because income distributions are often skewed; median is robust to extreme outliers.
   * Preserves customer sample size without dropping records.
2. **Record Dropping (`missing_strategy='drop'`)**:
   * Drops rows with missing required features (`subset=['annual_income', 'spending_score']`).
   * Cleanly resets DataFrame index and logs the number of dropped records.

### 6.3 Business Personas & Profiling
Post-clustering, `src/profiler.py` evaluates the unscaled cluster averages against overall population medians to categorize customer clusters into intuitive marketing segments:
* **High Income, High Spending**: *"High-Value VIPs"* (Prime targets for loyalty programs, premium product launches, and exclusive offers).
* **Low/Moderate Income, High Spending**: *"Carefree Spenders"* (Responsive to flash sales, trendy promotions, and financing/installment plans).
* **High Income, Low Spending**: *"Conservative Savers"* (Target with value propositions, quality assurances, and long-term warranties).
* **Low Income, Low Spending**: *"Budget-Conscious"* (Target with discount campaigns, coupons, and essential product bundles).
* **$K=1$**: *"Whole Customer Base (Unified Segment)"*.

---

## 7. Mandatory Test Cases Verification (TC1–TC6)

The test suite in `tests/test_customer_segmentation.py` rigorously validates all required test cases specified in the Problem Statement:

| Test Case | Condition / Input | Expected System Behavior | Verification Status |
| :--- | :--- | :--- | :---: |
| **TC1** | Dataset with clearly separated low-, medium-, and high-spending groups | Produces clusters that cleanly reflect these separated groups | **PASS** |
| **TC2** | User inputs $K = 3$ | Every valid customer is mapped to one of 3 cluster IDs (`{0, 1, 2}`) | **PASS** |
| **TC3** | User inputs $K = 1$ | Produces a single unified cluster (`cluster_id == 0`) without crashing | **PASS** |
| **TC4** | User inputs $K = 0$, negative values, or non-integers | Rejects invalid input immediately with an informative `ValueError` | **PASS** |
| **TC5** | Missing values (e.g. null income for customer) | Gracefully executes documented strategy (median imputation or drop) | **PASS** |
| **TC6** | Records count $N < K$ (fewer records than clusters) | Catches configuration gracefully with a descriptive `ValueError` | **PASS** |

### Running the Test Suite
Execute all tests with detailed verbosity:
```powershell
pytest -v
```

---

## 8. Sample Console Output

```text
================================================================================
CUSTOMER SEGMENTATION MACHINE LEARNING PIPELINE
================================================================================
Dataset path     : data/synthetic_customers.csv
Target clusters K: 3
Missing strategy : impute
Output directory : output
================================================================================

[1/5] Loading and validating dataset...
  -> Total records loaded: 200
  -> Valid records ready for analysis: 200

[2/5] Standardizing behavioral features with StandardScaler...
  -> Features scaled: ['annual_income', 'spending_score']
  -> Feature matrix shape: (200, 2)

[3/5] Training K-Means clustering algorithm (K=3)...
  -> Model converged with Inertia: 138.78
  -> Cluster distribution:
     * Cluster 0: 42 customers (21.0%)
     * Cluster 1: 118 customers (59.0%)
     * Cluster 2: 40 customers (20.0%)

[4/5] Computing cluster metrics and business personas...
================================================================================
CUSTOMER SEGMENTATION BUSINESS PROFILE SUMMARY
================================================================================

--- Cluster 0: Conservative Savers (High Income, Low Spending) ---
  * Size: 42 customers (21.0% of total)
  * Annual Income: Mean=$90,173.81 | Median=$89,600.00 | Range=[$65,800.00 - $114,600.00]
  * Spending Score: Mean=23.6 | Median=23.5 | Range=[12.0 - 48.0]
  * Purchase Frequency: Mean=3.9/yr

--- Cluster 1: Carefree Spenders (Low/Moderate Income, High Spending) ---
  * Size: 118 customers (59.0% of total)
  * Annual Income: Mean=$34,784.75 | Median=$27,850.00 | Range=[$15,200.00 - $68,400.00]
  * Spending Score: Mean=50.4 | Median=50.0 | Range=[4.0 - 96.0]
  * Purchase Frequency: Mean=5.4/yr

--- Cluster 2: High-Value VIPs (High Income, High Spending) ---
  * Size: 40 customers (20.0% of total)
  * Annual Income: Mean=$92,020.00 | Median=$91,550.00 | Range=[$73,200.00 - $112,800.00]
  * Spending Score: Mean=86.3 | Median=87.0 | Range=[59.0 - 99.0]
  * Purchase Frequency: Mean=12.5/yr

================================================================================

[5/5] Exporting outputs and generating visualizations...
  -> Exported segmented CSV: C:\Users\ranja\Downloads\games\detrax gaming\output\segmented_customers.csv
  -> Saved cluster scatter plot: C:\Users\ranja\Downloads\games\detrax gaming\output\customer_clusters_k3.png
  -> Saved elbow curve plot: C:\Users\ranja\Downloads\games\detrax gaming\output\elbow_curve.png

================================================================================
PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
```
