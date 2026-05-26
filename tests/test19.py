import sys
import os

# paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# paths
data_csv = os.path.join(PROJECT_ROOT, "data/raw/A_K_aukstais_udens.csv")
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
from tests.test_params.test19_params import params

# mainīgie
time_column_name = 'timestamp_unix'
value_column_name = 'value'
datetime_column_name = 'ts'


#Datu sagatavošana
prepared_data = prepare_data(data_csv, time_column_name, value_column_name, datetime_column_name)
preprocessed_data = preprocess_data(prepared_data, time_column_name, value_column_name, datetime_column_name)
days_df = day_use_dataset(preprocessed_data)

prepared_data = prepare_data(
    data_csv,
    time_column_name,
    value_column_name,
    datetime_column_name
)

preprocessed_data = preprocess_data(
    prepared_data,
    time_column_name,
    value_column_name,
    datetime_column_name
)

days_df = day_use_dataset(preprocessed_data)


threshold_array = dspot_array(
    days_df,
    params,
    value_column="delta"
)

labeled_df = create_dspot_labeled_dataset(days_df, threshold_array)

fig_path = os.path.join(FIG_DIR, "test19_dspot.png")

visualize_dspot_df(
    df=labeled_df,
    title=(
        f"Test 19 | "
        f"num_init={params['num_init']}, "
        f"depth={params['depth']}, "
        f"init_level={params['init_level']}, "
        f"risk={params['risk']}"
    ),
    save_path=fig_path
)

csv_path = os.path.join(CSV_DIR, "test19.csv")
labeled_df.to_csv(csv_path, index=False)

print("DONE")
print(fig_path)
print(csv_path)