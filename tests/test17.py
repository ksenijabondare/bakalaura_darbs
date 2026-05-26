import sys
import os
import pandas as pd

# paths
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

from src.prepare_data import (
    prepare_data,
    preprocess_data,
    day_use_dataset
)
from src.anomaly_detection_algorithms.dspot import (
    dspot_array
)

from src.visualizations import (
    visualize_dspot_df
)
from src.labeling_dataset import (
    create_dspot_labeled_dataset
)
from tests.test_params.test17_params import params



#Datu sagatavošana

days_df = pd.read_csv(data_csv)


threshold_array = dspot_array(
    days_df,
    params,
    value_column="delta"
)

labeled_df = create_dspot_labeled_dataset(days_df, threshold_array)


fig_path = os.path.join(FIG_DIR, "test17_dspot.png")

visualize_dspot_df(
    df=labeled_df,
    title=(
        f"Test 17 | "
        f"num_init={params['num_init']}, "
        f"depth={params['depth']}, "
        f"init_level={params['init_level']}, "
        f"risk={params['risk']}"
    ),
    save_path=fig_path
)


csv_path = os.path.join(CSV_DIR, "test17.csv")
labeled_df.to_csv(csv_path, index=False)

print("DONE")
print(fig_path)
print(labeled_df)