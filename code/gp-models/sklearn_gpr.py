'''
Sklearn GPR model training.
'''
from multiprocessing import pool
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time
import matplotlib.pyplot as plt
import os

def load_data(pollutant, data_path="data-collection/", timestep=None, subset=None, season=None, dayOfWeek=None, timeOfDay=None):
    """
    :param pollutant: {"CO", "NO2", "O3", "SO2", "PM10", "PM25"}
    :param data_path: path to data directory
    :param timestep: {"H", "D", "M", "Y"}
    :param subset: if provided, get data after the datetime
    :param season: if provided, {"winter", "spring", "summer", "autumn"} for seasonal buckets
            Winter: December, January, Ferbruary
            Spring: March, April and May
            Summer: June, July and August
            Autumn: September, October and November
    :param dayOfWeek: if provided, {"weekday", "weekend"}
    :param timeOfDay: if provided, {"day", "night"}
    """
    df = pd.read_csv(f"{data_path}{pollutant}.csv", parse_dates=["Timestamp"]).set_index("Timestamp")
    print(df)
    print(type(df.index))

    # daily, monthly, yearly
    if timestep in {"D", "M", "Y"}:
        index_format = {"D": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
        df = df.groupby(by=["SiteCode"]).resample(timestep).mean().dropna().reset_index()
        # df["date"] = df["date"].apply(lambda x: x.strftime(index_format[timestep]))

    if subset:
        df = df.loc[df["Timestamp"] > subset]

    # seasonal buckets
    if season:
        months = [12, 1, 2] if season == "winter" \
            else [i for i in range(3, 6)] if season == "spring" \
                else [i for i in range(6, 9)] if season == "summer" \
                    else [i for i in range(9, 12)] if season == "autumn" else [i for i in range(12)]

        df["Month"] = pd.DatetimeIndex(df['Timestamp']).month
        # print(df)

        # define condition: month must be within seasonal month range
        condition = ((df.Month >= months[0]) & (df.Month <= months[-1]))
        if season == "winter":
            condition = ((df.Month >= months[0]) | (df.Month <= months[-1]))
        df = df.loc[condition]
        
        print(df)
        print("Season: ", season, " , ", months)
        print("Season Months: ", df.Month.unique())

        # drop created month column
        df = df.drop(["Month"], axis=1)
        print(df)
        
    # day of week buckets: weekday vs weekend
    if dayOfWeek:
        days = [i for i in range(5)] if dayOfWeek == "weekday" else [i for i in range(5,7)]

        df["DayOfWeek"] = pd.DatetimeIndex(df['Timestamp']).dayofweek

        condition = ((df.DayOfWeek >= days[0]) & (df.DayOfWeek <= days[-1]))
        df = df.loc[condition]

        # print(df)
        # print("Day of week: ", dayOfWeek, " , ", days)
        # print("Day of week days: ", df.DayOfWeek.unique())

        df = df.drop(["DayOfWeek"], axis=1)
    
    # daytime (7am - 5pm) vs nighttime (5pm - 7am) buckets, according to London's sunrise and sunset times
    if timeOfDay:
        hours = [i for i in range(7, 18)] if timeOfDay == "day" else [i%24 for i in range(18, 31)]

        df["Hour"] = pd.DatetimeIndex(df['Timestamp']).hour

        condition = ((df.Hour >= hours[0]) & (df.Hour <= hours[-1]))
        if timeOfDay == "night":
            condition = ((df.Hour >= hours[0]) | (df.Hour <= hours[-1]))
        df = df.loc[condition]

        # print(df)
        # print("Time of day: ", timeOfDay, " , ", hours)
        # print("Hours of time of day in df: ", df.Hour.unique())

        df = df.drop(["Hour"], axis=1)

    return df

def transform_data(df, pollutant):
    N_MONTHS = 12
    N_DAYS = 30
    N_HOURS = 24
    start_year = df["Timestamp"].min().year
    df['t'] = df.apply(lambda row: (row.Timestamp.year-start_year)*N_MONTHS*N_DAYS*N_HOURS + (row.Timestamp.month%N_MONTHS)*N_DAYS*N_HOURS \
        + (row.Timestamp.day%N_DAYS)*N_HOURS + (row.Timestamp.hour%N_HOURS), axis=1)
    df = df.sort_values("t")

    lat_scaler = StandardScaler()
    df["norm_lat"] = lat_scaler.fit_transform(df[["Latitude"]].values)
    lon_scaler = StandardScaler()
    df["norm_lon"] = lon_scaler.fit_transform(df[["Longitude"]].values)
    pol_scaler = StandardScaler()
    df[f"norm_{pollutant}"] = pol_scaler.fit_transform(df[[pollutant]].values)

    return df, {"lat": lat_scaler, "lon": lon_scaler, pollutant: pol_scaler}

def get_X_Y_split(df, pollutant):
    X = df[["norm_lat", "norm_lon", "t"]].values
    Y = df[[f"norm_{pollutant}"]].values
    N, M = X.shape
    indices = np.arange(N)
    X_train, X_val, Y_train, Y_val, train_indices, test_indices = train_test_split(X, Y, indices, test_size=0.20)
    return X, X_train, X_val, Y, Y_train, Y_val, train_indices, test_indices

def plot_predictions(model, X, X_train, X_val, y, y_train, y_val, codes, title, save_path):
    '''
    Plots predictions before and after training GPR model for given site code sensor location
    :param site_code: str label of sensor location
    '''
    
    noise_std = 0.75

    # train_X, train_Y, test_X, test_Y = get_site_train_test_data(code)
    for code in codes:
        if (X_val.shape[0] > 0): # only plot if we have test data
            plt.figure(figsize=(12, 4))
            
            mean_prediction, std_prediction = model.predict(X, return_std=True) # Predict Y values at test locations
            X_val_t = X_val[:, 2] # only plot time for a given site
            # print("Validation X:")
            # print(X_val_t)

            train_X_t = X_train[:, 2] # only plot time for a given site
            # print("Train X:")
            # print(train_X_t)

            plt.plot(train_X_t, y_train, "x", label="Training points", alpha=0.2, color="green")

            X_val_t, mean_prediction = zip(*sorted(zip(X_val_t, mean_prediction)))
            (line,) = plt.plot(X_val_t, mean_prediction, lw=1.5, label="Mean of predictive posterior")
            col = line.get_color() 
            
            plt.errorbar(
                train_X_t,
                y_train,
                noise_std,
                linestyle="None",
                color="tab:blue",
                marker=".",
                markersize=10,
                label="Observations",
            )
            plt.fill_between(
                X_val_t,
                # (mean_prediction - 2 * std_prediction ** 0.5)[:, 0],
                # (mean_prediction + 2 * std_prediction ** 0.5)[:, 0],
                (mean_prediction - 1.96 * std_prediction)[:, 0],
                (mean_prediction + 1.96 * std_prediction)[:, 0],
                color=col,
                alpha=0.5,
                label=r"95% confidence interval"
            )
            # Z = model.inducing_variable.Z.numpy()
            # plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
            plt.legend(loc="lower right")
            
            # loss = model.training_loss((X_val, y_val))
            # loss = 0
            # score = model.score(X_val_t, mean_prediction)
            score = 0
            plt.title(f"{title} ({code}); val score (n={X_val.shape[0]})={score}")
            
            # create folder for model run
            isExist = os.path.exists(save_path)
            if not isExist:
                os.makedirs(save_path)

            plt.savefig(fname=save_path + "prediction_" + code + ".png")
            plt.close()



if __name__ == "__main__":
    # parameters
    pollutant = "NO2"
    timestep = "D"
    subset = datetime(2021, 1, 1)
    season = "summer"
    dayOfWeek = "weekday"
    timeOfDay = None # "day" # timestep must be "H" in order to create time of day buckets
    figure_title = pollutant + "_" + timestep + "_" + str(subset) + "_" + season + "_" + dayOfWeek


    print("Getting data...")
    # change subset param for all months
    monthly_NO2_df = load_data(pollutant, data_path="code/data-collection/LAQN_API_data/", timestep=timestep, subset=subset, season=season, dayOfWeek=dayOfWeek, timeOfDay=timeOfDay)

    print("Preparing data...")
    monthly_NO2_df, scalers = transform_data(monthly_NO2_df, pollutant)
    X, X_train, X_val, Y, Y_train, Y_val, train_indices, test_indices = get_X_Y_split(monthly_NO2_df, pollutant)
    site_codes = monthly_NO2_df["SiteCode"].unique()

    print("Creating model...")
    kernel = kernels.ExpSineSquared(periodicity=12)*kernels.RBF([0, 0, 1.0]) + kernels.RBF([1.0, 1.0, 0])
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)

    # plot_predictions(model, X, X_train, X_val, Y, Y_train, Y_val, site_codes, title="Predictions before training", save_path="code/gp-models/GPR_figures/"+ figure_title + "/preTraining/")

    print("Training model...")
    start = time.process_time()
    print(f"Started at {start}")
    model.fit(X_train, Y_train)
    model.fit(X_train, Y_train)
    elapsed_time = time.process_time() - start
    print(f"Finished at {elapsed_time}")

    # plot results
    plot_predictions(model, X, X_train, X_val, Y, Y_train, Y_val, site_codes, title="Predictions after training", save_path="code/gp-models/GPR_figures/" + figure_title + "/postTraining/")

    print("Scoring model...")
    model.score(X_val, Y_val)
    #final_train_X, final_train_Y, final_test_X, final_test_Y = get_site_train_test_data()

    print("Saving model...")
    filename = "code/gp-models/sklearn_GPR_no2_monthly.sav"
    pickle.dump(model, open(filename, "wb"))
    print("Successfully trained and saved model.")