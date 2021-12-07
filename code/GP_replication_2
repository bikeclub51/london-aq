from datetime import time
import gpflow
from gpflow.utilities import print_summary
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  



def get_data_df(pollutant, time_granularity="monthly"):
    '''
    Reads in data of pollutant concentrations from downloaded excel files from LAQN API and returns as a dataframe,
    with columns: SiteCode, Timestamp, pollutant, latitude, longitude, date.

    param pollutant : str : one of the following list of pollutant: "NO2", "CO", "O3", "PM10", "PM25", "SO2"
    param time_granularity : str : either "hourly", "daily", "monthly", "yearly"
    '''
    hourly_df = pd.read_csv("../data/" + pollutant + "_data.csv", sep=',', parse_dates=["Timestamp"]).set_index("Timestamp")
    if time_granularity == "hourly":
        return hourly_df
    
    elif time_granularity == "daily":
        daily_df = hourly_df.groupby(by=["SiteCode"]).resample("D", convention="start").mean().reset_index()
        daily_df["date"] = daily_df["Timestamp"].apply(lambda x: x.strftime("%Y"))
        return daily_df

    elif time_granularity == "monthly":
        monthly_df = hourly_df.groupby(by=["SiteCode"]).resample("M", convention="start").mean().reset_index()
        monthly_df["date"] = monthly_df["Timestamp"].apply(lambda x: x.strftime("%Y-%m"))
        monthly_df.set_index("date", inplace=True)
        return monthly_df

    elif time_granularity == "yearly":
        yearly_df = hourly_df.groupby(by=["SiteCode"]).resample("Y", convention="start").mean().reset_index()
        yearly_df["date"] = yearly_df["Timestamp"].apply(lambda x: x.strftime("%Y"))
        return yearly_df
    

def define_training_data(pollutant, df):
    ## divide into features and variable
    params = ['date', 'Latitude', 'Longitude']

    X = df[params].values
    y = df.loc[:, pollutant].values
    y = y.reshape(-1,1)

    ## print previews
    print("Model Features:")
    print(X.shape)
    print(X[1:10,:])

    print("--------------------")

    print("Variables: ", pollutant)
    print(y.shape)
    print(y[0:10])

    ## create validation dataset (no test set since using MLL)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0) 

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    feature_scaler = StandardScaler() 

    ## Normalize Y (after splitting into training and validation)
    ## standardize y-values
    y_train = feature_scaler.fit_transform(y_train)
    y_val = feature_scaler.fit_transform(y_val)

    return X_train, X_val, y_train, y_val


def define_kernels():
    k = gpflow.kernels.Constant()
    print_summary(k, fmt="notebook")
    return k


def build_gp_model(X_train, y_train, kernel):

    ## build model
    model = gpflow.models.GPR(X_train, y_train, kern=kernel)
    model.likelihood.variance = 0.01

    ## view 
    model.as_pandas_table()
    print(model.as_pandas_table())


if __name__ == '__main__':
    pollutant = "03"
    pollutant_df = get_data_df(pollutant, "daily")
    X_train, X_val, y_train, y_val = define_training_data(pollutant, pollutant_df)

    kernel = define_kernels()
    build_gp_model(X_train, y_train, kernel)
