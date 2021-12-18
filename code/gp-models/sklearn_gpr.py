'''
Sklearn GPR model training.
'''
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time

def load_data(pollutant, data_path="data/", timestep=None, subset=None):
    """
    :param pollutant: {"CO", "NO2", "O3", "SO2", "PM10", "PM25"}
    :param data_path: path to data directory
    :param timestep: {"H", "D", "M", "Y"}
    :param subset: if provided, get data after the datetime
    """
    df = pd.read_csv(f"{data_path}{pollutant}.csv", parse_dates=["date"]).set_index("date")
    if timestep in {"D", "M", "Y"}:
        index_format = {"D": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
        df = df.groupby(by=["code"]).resample(timestep).mean().dropna().reset_index()
        # df["date"] = df["date"].apply(lambda x: x.strftime(index_format[timestep]))
    
    if subset:
        df = df.loc[df["date"] > subset]
    return df

def transform_data(df, pollutant):
    N_MONTHS = 12
    start_year = df["date"].min().year
    df['t'] = df.apply(lambda row: (row.date.year-start_year)*N_MONTHS + (row.date.month%N_MONTHS), axis=1)
    df = df.sort_values("t")

    lat_scaler = StandardScaler()
    df["norm_lat"] = lat_scaler.fit_transform(df[["latitude"]].values)
    lon_scaler = StandardScaler()
    df["norm_lon"] = lon_scaler.fit_transform(df[["longitude"]].values)
    pol_scaler = StandardScaler()
    df[f"norm_{pollutant}"] = pol_scaler.fit_transform(df[[pollutant]].values)

    return df, {"lat": lat_scaler, "lon": lon_scaler, pollutant: pol_scaler}

def get_X_Y_split(df, pollutant):
    X = df[["norm_lat", "norm_lon", "t"]].values
    Y = df[[f"norm_{pollutant}"]].values
    N, M = X.shape
    indices = np.arange(N)
    X_train, X_val, Y_train, Y_val, train_indices, test_indices = train_test_split(X, Y, indices, test_size=0.20)
    return X_train, X_val, Y_train, Y_val, train_indices, test_indices

if __name__ == "__main__":
    print("Getting data...")
    monthly_NO2_df = load_data("NO2", data_path="../../data/", timestep="M", subset=datetime(2020, 1, 1))

    print("Preparing data...")
    monthly_NO2_df, scalers = transform_data(monthly_NO2_df, "no2")
    X_train, X_val, Y_train, Y_val, train_indices, test_indices = get_X_Y_split(monthly_NO2_df, "no2")

    print("Creating model...")
    kernel = kernels.ExpSineSquared(periodicity=12)*kernels.RBF([0, 0, 1.0]) + kernels.RBF([1.0, 1.0, 0])
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)

    print("Training model...")
    start = time.process_time()
    print(f"Started at {start}")
    model.fit(X_train, Y_train)
    elapsed_time = time.process_time() - start
    print(f"Finished at {elapsed_time}")

    print("Scoring model...")
    model.score(X_val, Y_val)
    #final_train_X, final_train_Y, final_test_X, final_test_Y = get_site_train_test_data()

    print("Saving model...")
    filename = "sklearn_GPR_no2_monthly.sav"
    pickle.dump(model, open(filename, "wb"))
    print("Successfully trained and saved model.")