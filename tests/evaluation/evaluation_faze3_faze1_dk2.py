import pandas as pd
import sys
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, PROJECT_ROOT)

pred_dir = os.path.join(
    PROJECT_ROOT,
    "test_outputs/anomaly_labeled_datasets"
)

# Testu pāri
test_pairs = [
    (7, 19),
    (8, 20),
    (9, 21),
]

results = []

for t1, t2 in test_pairs:

    true_df = pd.read_csv(os.path.join(pred_dir, f"test{t1}.csv"))
    pred_df = pd.read_csv(os.path.join(pred_dir, f"test{t2}.csv"))

    merged = true_df.merge(
        pred_df,
        on=["date", "delta"],
        suffixes=("_true", "_pred")
    )

    y_true = merged["over_threshold_true"].astype(int)
    y_pred = merged["over_threshold_pred"].astype(int)

    results.append({
        "test_pair": f"{t1}-{t2}",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    })

# FINAL DATAFRAME
evaluation_df = pd.DataFrame(results)

evaluation_dir = os.path.join(
    PROJECT_ROOT,
    "test_outputs/evaluation"
)

os.makedirs(evaluation_dir, exist_ok=True)

# Output file
evaluation_csv = os.path.join(
    evaluation_dir,
    "evaluation_results_test_7_9_19_21.csv"
)

# Export dataframe
evaluation_df.to_csv(evaluation_csv, index=False)
# Print result
print(evaluation_df)

import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

if np.array_equal(arr1, arr2):
    print("Equal")
else:
    print("Not Equal")