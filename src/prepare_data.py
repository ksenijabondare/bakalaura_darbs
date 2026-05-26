import pandas as pd
import os
from pathlib import Path



#Datu ielāde
def prepare_data(data_csv, time_column_name, value_column_name, datetime_column_name):
    df = pd.read_csv(data_csv)
    prepared_data = df[[time_column_name, value_column_name,datetime_column_name]].copy()

    prepared_data[datetime_column_name] = pd.to_datetime(prepared_data[datetime_column_name])
    prepared_data['date'] = prepared_data[datetime_column_name].dt.date #strftime("%d.%m.%Y")
    prepared_data['time'] = prepared_data[datetime_column_name].dt.time #strftime("%H:%M:%S")
    prepared_data[value_column_name] = pd.to_numeric(prepared_data[value_column_name], errors="coerce")

    return (prepared_data)


#Datu priekšapstrāde
def preprocess_data(prepared_data, time_column_name, value_column_name, datetime_column_name):
    preprocessed_data = prepared_data.copy()
    preprocessed_data[value_column_name] = preprocessed_data[value_column_name].interpolate(method='linear')
    preprocessed_data['delta'] = preprocessed_data[value_column_name].diff()
    return preprocessed_data

#Datu kopa ar dienas patēriņu
def day_use_dataset(preprocessed_data):
    preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'], format='%Y:%m:%d')
    days_df = (preprocessed_data.groupby('date', as_index=False)['delta'].sum(min_count=1)  )
    return days_df

def month_analysis(days_df, month_number, date_column="date"):
    """
    Sagatavo datu kopu līdz izvēlētajam mēnesim ieskaitot.

    Parameters:
        days_df (pd.DataFrame): dienu patēriņa datu kopa
        month_number (int): mēneša numurs (1-12)
        date_column (str): datuma kolonnas nosaukums

    Returns:
        pd.DataFrame: filtrēta datu kopa
    """

    df = days_df.copy()

    # Pārliecinās, ka datuma kolonna ir datetime formātā
    df[date_column] = pd.to_datetime(df[date_column])

    # Atlasa visus datus līdz izvēlētajam mēnesim ieskaitot
    filtered_df = df[df[date_column].dt.month <= month_number]

    return filtered_df