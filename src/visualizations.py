import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_pot(
    data: np.ndarray,
    threshold: float,
    init_threshold: float,
    title: str = "POT Anomaly Detection"
):


    fig, ax = plt.subplots(1, figsize=(10, 3))


    ax.plot(
        data,
        label="Data",
        color=cm.Set1(1),
        lw=1
    )

    ax.axhline(
        init_threshold,
        color=cm.Set1(8),
        ls="-",
        lw=1,
        label="Initial threshold"
    )


    ax.axhline(
        threshold,
        color=cm.Set1(0),
        ls="-",
        lw=1,
        label="Threshold"
    )

    mask = data > threshold

    ax.scatter(
        np.arange(len(data))[mask],
        data[mask],
        facecolor="none",
        edgecolor=cm.Set1(0),
        s=20,
        label="Outliers"
    )


    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='center left',
        ncol=4,
        bbox_to_anchor=(0.26, -0.05)
    )

    ax.grid(
        axis='both',
        color='black',
        alpha=0.1
    )

    ax.set_xlim(0, len(data))

    ax.set_title(title)

    plt.tight_layout()

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd
import matplotlib.pyplot as plt

def visualize_paterins(
    df,
    date_column="date",
    value_column="delta",
    anomaly_column=None
):


    data = df.copy()

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column)

    plt.figure(figsize=(12, 5))

    plt.plot(data[date_column], data[value_column], label="Patēriņš", linewidth=2)

    if anomaly_column and anomaly_column in data.columns:
        anomalies = data[data[anomaly_column] == True]
        plt.scatter(
            anomalies[date_column],
            anomalies[value_column],
            color="red",
            label="Anomālija"
        )

    plt.title("Patēriņa dinamika laika gaitā")
    plt.xlabel("Datums")
    plt.ylabel("Patēriņš")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def visualize_anomalies_over_threshold(
    df,
    date_column="date",
    value_column="delta",
    anomaly_column="over_threshold"
):


    data = df.copy()

    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(date_column)

    plt.figure(figsize=(12, 5))

    plt.plot(
        data[date_column],
        data[value_column],
        label="Delta",
        linewidth=2
    )

    anomalies = data[data[anomaly_column] == True]

    plt.scatter(
        anomalies[date_column],
        anomalies[value_column],
        color="red",
        label="Anomaly (over threshold)",
        zorder=5
    )

    plt.title("Anomalijas (over threshold) laika rindā")
    plt.xlabel("Datums")
    plt.ylabel("Delta")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def visualize_dspot_test(
    data: np.ndarray,
    threshold: np.ndarray,
    num_init: int,
    depth: int,
    title: str = "DSPOT Anomaly Detection",
    save_path: str = None, 
    show: bool = True    
):

    
    fig, ax = plt.subplots(1, figsize=(10, 3))
    

    ax.plot(
        data,
        label="Data",
        color=plt.cm.Set1(1),
        lw=1
    )
    
    start = num_init + depth
    

    ax.plot(
        threshold[:start],
        color="gray",
        ls="--",
        lw=1,
        label="Initial threshold"
    )

    ax.plot(
        np.arange(start, len(data)),
        threshold[start:],
        color=plt.cm.Set1(0),
        lw=1,
        label="Threshold"
    )
    

    mask = data > threshold
    
    ax.scatter(
        np.arange(len(data))[mask],
        data[mask],
        facecolor="none",
        edgecolor=plt.cm.Set1(0),
        s=20,
        label="Outliers"
    )

    handles, labels = ax.get_legend_handles_labels()
    
    fig.legend(
        handles,
        labels,
        loc='center left',
        ncol=4,
        bbox_to_anchor=(0.26, -0.05)
    )
    
    ax.grid(
        axis='both',
        color='black',
        alpha=0.1
    )
    
    ax.set_xlim(0, len(data))
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figūra saglabāta: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def visualize_dspot_month(
    dataset,
    threshold_array,
    month_number,
    date_column="date",
    value_column="delta",
    title="DSPOT Monthly Anomaly Detection",
    save_path=None,
    show=True
):
    """
    Vizualizē tikai izvēlētā mēneša DSPOT rezultātus.

"""

    df = dataset.copy()

    if len(df) != len(threshold_array):
        raise ValueError("Dataset and threshold array must have same length")

    df[date_column] = pd.to_datetime(df[date_column])

    df["threshold"] = np.array(threshold_array)

    month_df = df[df[date_column].dt.month == month_number].copy()

    if month_df.empty:
        raise ValueError(f"No data found for month {month_number}")

    # Anomāliju maska
    anomaly_mask = month_df[value_column] > month_df["threshold"]

    fig, ax = plt.subplots(1, figsize=(12, 4))

    ax.plot(
        month_df[date_column],
        month_df[value_column],
        label="Data",
        lw=1
    )

    ax.plot(
        month_df[date_column],
        month_df["threshold"],
        label="Threshold",
        lw=1
    )

    ax.scatter(
        month_df.loc[anomaly_mask, date_column],
        month_df.loc[anomaly_mask, value_column],
        color="red",
        s=30,
        label="Anomalies"
    )

    ax.set_title(f"{title} - Month {month_number}")
    ax.set_xlabel("Date")
    ax.set_ylabel(value_column)

    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figūra saglabāta: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_dspot_df(
    df: pd.DataFrame,
    title: str = "DSPOT Anomaly Detection",
    save_path: str = None,
    show: bool = True,
    figsize: tuple = (12, 4)
):

    required_cols = ["date", "delta", "threshold", "over_threshold"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Trūkst kolonnas: {missing}")

    plot_df = df.copy()

    plot_df["date"] = pd.to_datetime(plot_df["date"])

    plot_df = plot_df.sort_values("date")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        plot_df["date"],
        plot_df["delta"],
        label="Delta",
        color=plt.cm.Set1(1),
        lw=1
    )

    ax.plot(
        plot_df["date"],
        plot_df["threshold"],
        label="Threshold",
        color=plt.cm.Set1(0),
        lw=1
    )

    anomalies = plot_df["over_threshold"].astype(bool)

    ax.scatter(
        plot_df.loc[anomalies, "date"],
        plot_df.loc[anomalies, "delta"],
        facecolor="none",
        edgecolor=plt.cm.Set1(0),
        s=30,
        label="Outliers"
    )

    ax.grid(
        axis="both",
        color="black",
        alpha=0.1
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Delta")

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.autofmt_xdate()

    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figūra saglabāta: {save_path}")


    if show:
        plt.show()
    else:
        plt.close()