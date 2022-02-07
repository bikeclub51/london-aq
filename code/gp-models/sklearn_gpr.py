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
import random

def load_data(pollutant, data_path="data/", timestep=None, cutoff=None, subset=None):
    """
    :param pollutant: {"CO", "NO2", "O3", "SO2", "PM10", "PM25"}
    :param data_path: path to data directory
    :param timestep: {"H", "D", "M", "Y"}
    :param subset: if provided, get data after the datetime
    """
    df = pd.read_csv(f"{data_path}{pollutant}.csv", parse_dates=["date"])
    
    if cutoff:
        df = df.loc[df["date"] > cutoff]

    if subset:
        df["month"] = df["date"].dt.strftime('%B')
        df = df.loc[df["month"].isin(subset)]

    if timestep in {"A", "D", "M", "Y"}:
        if timestep == "A":
            df = df.groupby(by=["code"]).mean().dropna().reset_index()
        else:
            index_format = {"D": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
            df = df.set_index("date")
            df = df.groupby(by=["code"]).resample(timestep).mean().dropna().reset_index()
    
    df = df.rename({"latitude": "Latitude", "longitude": "Longitude"}, axis="columns")
    return df

def transform_data(df, pollutant, time=False, normalize=True):
    if time:
        N_MONTHS = 12
        start_year = df["date"].min().year
        df['t'] = df.apply(lambda row: (row.date.year-start_year)*N_MONTHS + (row.date.month%N_MONTHS), axis=1)
        df = df.sort_values("t")

    if not normalize:
        return df, None

    lat_scaler = StandardScaler()
    df["norm_lat"] = lat_scaler.fit_transform(df[["Latitude"]].values)
    lon_scaler = StandardScaler()
    df["norm_lon"] = lon_scaler.fit_transform(df[["Longitude"]].values)
    pol_scaler = StandardScaler()
    df[f"norm_{pollutant}"] = pol_scaler.fit_transform(df[[pollutant]].values)
    return df, {"lat": lat_scaler, "lon": lon_scaler, pollutant: pol_scaler}


def get_X_Y_split(df, pollutant, time=False, normalized=True):
    lat = "Latitude" if not normalized else "norm_lat"
    lon = "Longitude" if not normalized else "norm_lon"
    pol = pollutant if not normalized else f"norm_{pol}"

    if time:
        X = df[[lat, lon, "t"]].values
        Y = df[[pol]].values
    else:
        X = df[[lat, lon]]
        Y = df[[pol]]

    N, M = X.shape
    indices = np.arange(N)
    X_train, X_val, Y_train, Y_val, train_indices, test_indices = train_test_split(X, Y, indices, test_size=0.20)
    return X_train, X_val, Y_train, Y_val, train_indices, test_indices

if __name__ == "__main__":
    print("Getting data...")
    sub = {"September", "October", "November", "December", "January", "February", "March"}
    df = load_data("NO2", data_path="../../data/", timestep="A", subset=sub)
    print(df.head())

    print("Preparing data...")
    df, scalers = transform_data(df, "no2", normalize=False)
    X_train, X_val, Y_train, Y_val, train_indices, test_indices = get_X_Y_split(df, "no2", normalized=False)
    print(X_train.shape)

    print("Creating model...")
    #kernel = kernels.ExpSineSquared(periodicity=12)*kernels.RBF([0, 0, 1.0]) + kernels.RBF([1.0, 1.0, 0])
    kernel = kernels.RBF([1.0, 1.0])
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)

    print("Training model...")
    start = time.process_time()
    print(f"Started at {start}")
    model.fit(X_train, Y_train)
    elapsed_time = time.process_time() - start
    print(f"Finished at {elapsed_time}")

    print("Scoring model...")
    y_mean, y_cov = model.predict(X_val, return_cov=True)
    print(X_val.shape)
    print(y_cov.shape)
    #final_train_X, final_train_Y, final_test_X, final_test_Y = get_site_train_test_data()

    print("Saving model...")
    random_id = random.randint(0, 10000)
    filename = f"{random_id}_no2_winter_months"
    df_filename = f"{filename}_df.sav"
    model_filename = f"{filename}_model.sav"
    pickle.dump(df, open(df_filename, "wb"))
    pickle.dump(model, open(model_filename, "wb"))
    print("Successfully trained and saved model at:")
    print(df_filename)
    print(model_filename)