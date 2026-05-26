import pandas as pd
import numpy as np

def artificial_leakage(dataset, start_date, end_date, leakage_scale):

    if not (0 <= leakage_scale <= 1):
        raise ValueError("leakage_scale must be between 0 and 1")

    df = dataset.copy()

    df["date"] = pd.to_datetime(df["date"])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    df.loc[mask, "delta"] = df.loc[mask, "delta"] * (1 + leakage_scale)

    return df

def extract_over_threshold_only(
    labeled_df,
    date_column="date",
    value_column="delta",
    label_column="over_threshold"
):

    required_cols = [date_column, value_column, label_column]
    missing = [c for c in required_cols if c not in labeled_df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return labeled_df[labeled_df[label_column] == True].copy()


def create_dspot_labeled_dataset(df, threshold_array):
    """
    Returns:
        columns: date, delta, threshold, over_threshold
    """

    out = df.copy()

    if len(out) != len(threshold_array):
        raise ValueError("Length mismatch between df and threshold_array")

    out["threshold"] = threshold_array

    out["over_threshold"] = (
        out["threshold"].notna() &
        (out["delta"] > out["threshold"])
    )

    return out[["date", "delta", "threshold", "over_threshold"]]
