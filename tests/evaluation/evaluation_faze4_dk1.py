import pandas as pd
import sys
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

# Load datasets

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

# mainīgie
# Paths
true_csv = os.path.join(
    PROJECT_ROOT,
    "data/labeled/labeled_df_leak_march.csv"
)

pred_dir = os.path.join(
    PROJECT_ROOT,
    "test_outputs/anomaly_labeled_datasets"
)

# Load ground truth once
true_df = pd.read_csv(true_csv)
true_df = true_df.rename(columns={"anomaly": "over_threshold"})
# Store evaluation results
evaluation_results = []

# Loop through test1-test3
for i in range(25, 27):

    test_name = f"test{i}"

    pred_csv = os.path.join(
        pred_dir,
        f"{test_name}.csv"
    )

    # Skip missing files
    if not os.path.exists(pred_csv):
        print(f"Missing file: {pred_csv}")
        continue

    # Load prediction dataframe
    pred_df = pd.read_csv(pred_csv)

    # Ensure rows align
    merged = true_df.merge(
        pred_df,
        on=["date", "delta"],
        suffixes=("_true", "_pred")
    )

    # Convert labels to integers
    y_true = merged["over_threshold_true"].astype(int)
    y_pred = merged["over_threshold_pred"].astype(int)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Save metrics
    evaluation_results.append({
        "test": test_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

# Create dataframe
evaluation_df = pd.DataFrame(evaluation_results)
# Create evaluation output directory
evaluation_dir = os.path.join(
    PROJECT_ROOT,
    "test_outputs/evaluation"
)

os.makedirs(evaluation_dir, exist_ok=True)

# Output file
evaluation_csv = os.path.join(
    evaluation_dir,
    "evaluation_results_test_25_26.csv"
)

# Export dataframe
evaluation_df.to_csv(evaluation_csv, index=False)
# Print result
print(evaluation_df)