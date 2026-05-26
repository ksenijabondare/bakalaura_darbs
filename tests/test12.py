import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# paths
data_csv = os.path.join(PROJECT_ROOT, "data/labeled/labeled_df_leak_march.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "test_outputs/test_params")
FIG_DIR = os.path.join(PROJECT_ROOT, "test_outputs/test_anomaly_visualizations")  # labots
CSV_DIR = os.path.join(PROJECT_ROOT, "test_outputs/anomaly_labeled_datasets")

# izveidot direktorijas
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# mainīgie
time_column_name = 'timestamp_unix'
value_column_name = 'value'
datetime_column_name = 'ts'

from src.prepare_data import (
    prepare_data,
    preprocess_data,
    day_use_dataset
)
from src.anomaly_detection_algorithms.dspot import (
    DriftStreamingPeakOverThreshold,
    create_dspot_labeled_dataset
)
from src.visualizations import (
    visualize_dspot_df
)
from tests.test_params.test12_params import params

# Datu sagatavošana
days_df = pd.read_csv(data_csv)
leaked_df = days_df.copy()

data = leaked_df["delta"].to_numpy()

# DSPOT parametri
threshold = DriftStreamingPeakOverThreshold(
    data=data,
    num_init=params["num_init"],
    depth=params["depth"],
    num_candidates=params["num_candidates"],
    risk=params["risk"],
    init_level=params["init_level"],
    epsilon=params["epsilon"],
)

days_df["threshold"] = threshold
days_df["over_threshold"] = days_df["delta"] > days_df["threshold"]

labeled_df = days_df[["date", "delta", "threshold", "over_threshold"]]

fig_path = os.path.join(FIG_DIR, "test12_dspot.png")

visualize_dspot_df(
    df=labeled_df,
    title=(
        f"Test 12 | "
        f"num_init={params['num_init']}, "
        f"depth={params['depth']}, "
        f"init_level={params['init_level']}, "
        f"risk={params['risk']}"
    ),
    save_path=fig_path
)
labeled_df = create_dspot_labeled_dataset(days_df, threshold)
csv_path = os.path.join(CSV_DIR, "test12.csv")
labeled_df.to_csv(csv_path, index=False)

print(f"Vizualizācija saglabāta: {fig_path}")
print(f"Anotētie dati saglabāti: {csv_path}")