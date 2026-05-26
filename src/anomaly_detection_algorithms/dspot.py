import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd


from src.anomaly_detection_algorithms.pot import PeakOverThreshold
from src.anomaly_detection_algorithms.pot import Grimshaw
from src.anomaly_detection_algorithms.pot import CalcThreshold


def DriftStreamingPeakOverThreshold(
    data: np.array, 
    num_init: int, 
    depth: int,
    num_candidates: int,
    risk: float = 1e-3, 
    init_level: float = 0.9, 
    epsilon: float = 1e-8,
):

    # ņem sākotnējo logu (slīdošais logs driftam)
    w = data[:depth].copy()
    # aprēķina sākotnējo vidējo vērtību
    m = w.mean()
    
    # saraksts, kur glabājam atlikumus
    x_ = []
    # sākotnējā modeļa mācīšana
    for i in range(depth, depth + num_init):
        # novirze no vidējā
        x_.append(data[i] - m)
        # atjauno vidējo vērtību no slīdošā loga
        start_idx = max(0, i - depth + 1)
        m = data[start_idx:i+1].mean()
    
    x_ = np.array(x_)
    
    # izveido sākotnējo slieksni ar pot algoritmu
    z, t = PeakOverThreshold(
        data=x_,
        num_candidates=num_candidates,
        risk=risk,
        init_level=init_level,
        epsilon=epsilon
    )
    
    # izvēlas ekstrēmās vērtības
    y = x_[x_ > t] - t
    
    # sākotnējie sliekšņi visiem pirmajiem datiem
    z_list = [z + w.mean()] * (depth + num_init)
    
    # apstrādā pārējos datus
    for i in range(depth + num_init, len(data)):
        # pašreizējais vidējais
        current_m = w.mean()
        # novirze no vidējā
        x_val = data[i] - current_m
        
        # pievieno vērtību vēsturei
        x_ = np.append(x_, x_val)
        # anomāliju pārbaude
        if x_val > z:
            z_list.append(z + current_m)
        else:
            # ja vērtība ir ekstrēma (bet ne anomālija), atjauno modeli
            if x_val > t:
                y = np.append(y, x_val - t)
                
                gamma, sigma = Grimshaw(
                    peaks=y,
                    threshold=t,
                    num_candidates=num_candidates,
                    epsilon=epsilon,
                )
                
                z = CalcThreshold(
                    q=risk,
                    gamma=gamma,
                    sigma=sigma,
                    n=y.size,
                    N=len(x_),
                    t=t,
                )
            
            z_list.append(z + current_m)
        
        # atjauno slīdošo logu (driftam)
        w = np.append(w[1:], data[i])
    
    return np.array(z_list)

def create_dspot_labeled_dataset(
    dataset,
    threshold_array,
    date_column="date",
    value_column="delta",
    label_column="over_threshold"
):

    df = dataset.copy()

    if len(df) != len(threshold_array):
        raise ValueError("Dataset and threshold array must have same length")

    df[date_column] = pd.to_datetime(df[date_column])

    thresholds = np.array(threshold_array)

    df[label_column] = df[value_column].values > thresholds

    return df


def dspot_array(df, params, value_column="delta"):
    """
    DSPOT tikai uz vērtībām virs 3*std filtrācijas,
    bet atgriež FULL length threshold array, kas ir orais algoritma variants.
    """

    series = df[value_column]

    mean = series.mean()
    std = series.std()

    lower_bound = mean - 3 * std

    valid_mask = (series > lower_bound) & (series.notna())

    df_filtered = df[valid_mask].copy()
    data_filtered = df_filtered[value_column].to_numpy()

    # DSPOT
    threshold_filtered = DriftStreamingPeakOverThreshold(
        data=data_filtered,
        num_init=params["num_init"],
        depth=params["depth"],
        num_candidates=params["num_candidates"],
        risk=params["risk"],
        init_level=params["init_level"],
        epsilon=params["epsilon"],
    )

    # pilns array
    threshold_full = np.full(len(df), np.nan)
    threshold_full[valid_mask.values] = threshold_filtered

    return threshold_full
