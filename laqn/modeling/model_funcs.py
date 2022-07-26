import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import random
import bayesnewton
from scipy.cluster.vq import kmeans2
from sklearn.model_selection import train_test_split
import objax
import time
from typing import List, Set, Tuple

def load_data(pollutant:str, data_dir:str="data/", time_step:str=None, time_range:Tuple[datetime, datetime]=None,
            season:bool=None, day_of_week:bool=None, time_of_day:bool=None, decimals:int=None, cutoff:float=0) -> pd.DataFrame:
    """
    Loads the requested data into a pandas DataFrame.

    Args:
        pollutant: {"CO", "NO2", "O3", "SO2", "PM10", "PM25"}
        data_dir: Path to data directory.
        time_step: Averages data to hourly, daily, monthly, or yearly bins. {"H", "D", "M", "Y"}
        time_range: Gets only data between time_range[0] and time_range[1] (or present day, if time_range[1] is not provided)
        season: If True, adds a binary season column for each season.
            December, January, Ferbruary -> Winter
            March, April, May -> Spring
            June, July, August -> Summer
            September, October, November - Autumn
        day_of_week: If True, adds a binary weekday vs. weekend column.
            weekday -> 0
            weekend -> 1
        time_of_day: If True, adds a binary daytime vs. nightime column.
            daytime (7am-5pm) -> 0
            nighttime (5pm-7am) -> 1
        decimals: Rounds the data to the given number of decimal places.
        cutoff: Float in the range (0, 1). Filters out any sites which do not have a data ratio higher than the given value over all time steps.

    Returns:
        A pandas DataFrame with columns ["date", pollutant(s), "site", "code", "latitude", "longitude", "site_type", "t"] where "t" is the time step of the data point.
        May also include columns ["winter", "spring", "summer", "autumn", "day_of_week", "time_of_day"].
    """
    print("[Data] Loading data...")
    data_t0 = time.time()
    df = pd.read_csv(f"{data_dir}{pollutant}.csv", parse_dates=["date"])

    df = df.loc[df[pollutant.lower()] > 0]
    # Get data within the given time range
    if time_range:
        start, end = time_range
        if not end:
            end = datetime.today
        df = df.loc[(df["date"] >= start) & (df["date"] <= end)]

    df = df.set_index("date")
    # Bin to given time step
    if time_step in {"D", "M", "Y"}:
        index_format = {"D": "%Y-%m-%d", "M": "%Y-%m", "Y": "%Y"}
        df = df.groupby(by=["code"]).resample(time_step).mean().dropna()
        # df["date"] = df["date"].apply(lambda x: x.strftime(index_format[time_step]))

    df = df.reset_index()
    # Seasonal buckets
    # Creates binary column for each season (1 represents that the data point is part of that season, 0 represents not part of season)
    if season:
        seasons = {"winter": [12, 1, 2], 
                "spring": [3, 4, 5], 
                "summer" : [6, 7, 8], 
                "autumn" : [9, 10, 11]}
        df["Month"] = pd.DatetimeIndex(df["date"]).month
        
        for s in seasons:
            condition = ((df.Month >= seasons[s][0]) & (df.Month <=seasons[s][-1]))
            if s == "winter":
                condition = ((df.Month >= seasons[s][0]) | (df.Month <= seasons[s][-1]))
            df[s] = np.where(condition, 1, 0)

        # Drop created month column
        df = df.drop(["Month"], axis=1)
        
    # Day of week buckets
    # Creates binary column for "day_of_week" (1 represents weekday, 0 represents weekend)
    if day_of_week:
        weekdays = [i for i in range(5)]
        df["DayOfWeek"] = pd.DatetimeIndex(df["date"]).dayofweek
        condition = ((df.DayOfWeek >= weekdays[0]) & (df.DayOfWeek <= weekdays[-1]))

        df["day_of_week"] = np.where(condition, 1, 0)
        df = df.drop(["DayOfWeek"], axis=1)
        
    
    # Daytime (7am - 5pm) vs nighttime (5pm - 7am) buckets, according to London"s sunrise and sunset times
    # Creates binary column for "time_of_day" (1 represents daytime, 0 represents nighttime)
    if time_of_day:
        daytime_hours = [i for i in range(7, 18)]
        df["Hour"] = pd.DatetimeIndex(df["date"]).hour

        condition = ((df.Hour >= daytime_hours[0]) & (df.Hour <= daytime_hours[-1]))
        if time_of_day == "night":
            condition = ((df.Hour >= daytime_hours[0]) | (df.Hour <= daytime_hours[-1]))

        df["time_of_day"] = np.where(condition, 1, 0)
        df = df.drop(["Hour"], axis=1)

    # Generate time_step (t) column
    df = df.sort_values("date")
    dates = df["date"].values

    t = -1
    current_date = None
    time_steps = []
    for date in dates:
        if date != current_date:
            t += 1
            current_date = date
        time_steps.append(t)
    df["t"] = time_steps
    
    # Round measurement values
    if decimals:
        df["latitude"] = df["latitude"].round(decimals)
        df["longitude"] = df["longitude"].round(decimals)
    
    # Filter out sites which don"t meet the cutoff
    t_max = df["t"].max()
    site_data_counts = np.array(np.unique(df["code"], return_counts=True)).T
    kept_sites = [site_data_counts[i, 0] for i in range(site_data_counts.shape[0]) if site_data_counts[i, 1]/(t_max+1) > cutoff]
    df = df.loc[df["code"].isin(kept_sites)]

    data_t1 = time.time()
    data_t = data_t1-data_t0    
    print(f"[Data] Time to load data: {data_t:.2f} s")
    return df

def get_data_split_sites(df:pd.DataFrame, num_val_sites:int=1, num_train_sites:int=70, val_sites:Set[str]=None, train_sites:Set[str]=None) -> Tuple[set, set, set]:
    """
    Categorizes the sites in the given DataFrame into training, test, and validation sites.

    The user should either set (num_val_sites and num_train_sites) or (val_sites and train_sites), but not both pairs.

    Args:
        df: LAQN pollutant measurement data pandas DataFrame.
        num_val_sites: Number of validation sites to randomly set.
        num_train_sites: Number of training sites to randomly set.
            The test sites are the remainder of these two sets.
        val_sites: Set of sites which should be set as the validation sites.
        train_sites: Set of sites which should be set as the validation sites.
            The test sites are the remainder of these two sites.

    Returns:
        A tuple consisting of the training, test, and validation site sets.
    """
    sites = set(df["code"].unique())
    if not val_sites:
        val_sites = set(random.sample(sites, num_val_sites))
    else:
        val_sites = set(val_sites)
    sites = sites - val_sites
    
    if not train_sites:
        train_sites = set(random.sample(sites, num_train_sites))
    else:
        train_sites = set(train_sites)
    test_sites = sites - train_sites
    return train_sites, test_sites, val_sites

def split_data_on_sites(df:pd.DataFrame, sites:Tuple[set, set, set]) -> pd.DataFrame:
    """
    Adds a column to the DataFrame representing the dataset each site belongs to.

    Args:
        df: LAQN pollutant measurement data pandas DataFrame.
        sites: A tuple consisting of the training, test, and validation site sets.
    
    Returns:
        A copy of the pandas DataFrame with a new "dataset" column which has a value in {"train", "test", "val"}.
    """
    train_sites, test_sites, val_sites = sites
    site_map = {}
    site_map.update({train_site: "train" for train_site in train_sites})
    site_map.update({test_site: "test" for test_site in test_sites})
    site_map.update({val_site: "val" for val_site in val_sites})
    df["dataset"] = df["code"].map(site_map)
    return df.copy()

def split_data_randomly(df:pd.DataFrame, test_size:float=0.2) -> pd.DataFrame:
    """
    Splits the data in the given pandas DataFrame randomly into train and test datasets.

    Adds a column representing the dataset each data point belongs to

    Args:
        df: LAQN pollutant measurement data pandas DataFrame.
        test_ratio: Float in the range (0, 1). Size of the test set.

    Returns:
        A copy of the pandas Dataframe with a new "dataset" column which has a value in {"train", "test"}.
    
    TODO: Add support for validation sites. See:
    https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
    """
    train_df, test_df = train_test_split(df, test_size=test_size)
    ind_map = {}
    ind_map.update({train_ind: "train" for train_ind in train_df.index})
    ind_map.update({test_ind: "test" for test_ind in test_df.index})
    df["dataset"] = df.index.map(ind_map)
    return df.copy()

def scale_data(df:pd.DataFrame, features:List[str], decimals=6):
    """
    Normalizes the data features in the DataFrame according to the values of the features in the training data.

    Args:
        df: LAQN pollutant measurement data pandas DataFrame.
        features: Dimensions to normalize.
        decimals: Rounds the data to the given number of decimal places.

    Returns:
        A copy of the pandas DataFrame with a new "scaled_f" column for each feature "f" in the given list of features along with a dict mapping
        the name of each feature "f" to its respective StandardScaler object.
    """
    scalers = {}
    for feature in features:
        scaler = StandardScaler()
        train_df = df.loc[df["dataset"] == "train"]
        train_vals = train_df[[feature]].values
        scaler.fit(train_vals)
        df[f"scaled_{feature}"] = df[feature].apply(lambda x: np.round_(scaler.transform(np.array([[x]])), decimals).item())
        scalers[feature] = scaler
    return df.copy(), scalers

def train_model(df:pd.DataFrame, x_features:List[str], y_features:List[str], sparse:bool=True, num_z_space:int=30) -> bayesnewton.models.MarkovVariationalGP:
    """
    Trains an ST-SVGP model on the given data with respect to the given X and Y features.
    
    Follows the default parameters set in the original ST-SVGP paper:
    https://proceedings.neurips.cc/paper/2021/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf
    https://github.com/AaltoML/spatio-temporal-GPs/blob/main/experiments/air_quality/models/m_bayes_newt.py

    Args:
        df: LAQN pollutant measurement data pandas DataFrame.
        x_features: Input dimension(s) (column(s) in df) to train on.
        y_features: Output dimension(s) (column(s) in df) to train against.
        sparse: If True, runs a k-means algorithm to set inducing points.
        num_z_space: Number of inducing points to set.

    Returns:
        A trained ST-SVGP model.
    """
    print(f"[ST-SVGP] Starting model training...")
    train_df = df.loc[df["dataset"] == "train"]
    X = train_df[x_features].values
    Y = train_df[y_features].values
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    Nt = t.shape[0]
    print(f"[ST-SVGP] Num time steps: {Nt}")
    Nr = R.shape[1]
    print(f"[ST-SVGP] Num spatial points: {Nr}")
    N = Nt * Nr
    print(f"[ST-SVGP] Num data points {N}")

    var_y = 5.
    var_f = 1.
    len_time = 0.01
    len_space = 0.2

    opt_z = sparse

    if sparse:
        z = kmeans2(R[0, ...], num_z_space, minit="points")[0]
    else:
        z = R[0, ...]

    kern_time = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_time)
    kern_space0 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    kern_space1 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
    kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

    kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                                    spatial_kernel=kern_space,
                                                    z=z,
                                                    sparse=sparse,
                                                    opt_z=opt_z,
                                                    conditional="Full")

    lik = bayesnewton.likelihoods.Gaussian(variance=var_y)

    model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=False)

    lr_adam = 0.01
    lr_newton = 1.
    iters = 300
    opt_hypers = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())


    @objax.Function.with_vars(model.vars() + opt_hypers.vars())
    def train_op():
        model.inference(lr=lr_newton)  # perform inference and update variational params
        dE, E = energy()  # compute energy and its gradients w.r.t. hypers
        opt_hypers(lr_adam, dE)
        return E

    train_op = objax.Jit(train_op)
    
    train_t0 = time.time()
    for i in range(1, iters + 1):
        loss = train_op()
        print("iter %2d: energy: %1.4f" % (i, loss[0]))
    train_t1 = time.time()
    train_t = train_t1-train_t0
    print(f"[ST-SVGP] Train time: {train_t:.2f} s")
    avg_iter_t = train_t/iters
    print(f"[ST-SVGP] Average iter time: {avg_iter_t:.2f}")

    return model

def eval_model(GP:bayesnewton.models.MarkovVariationalGP, df:pd.DataFrame, x_features:list, y_features:list, dataset:str="val") -> Tuple[float, float]:
    """
    Calculates the NLPD and RMSE of the given model on the given dataset category.

    Args:
        GP: A trained ST-SVGP model.
        df: The LAQN pollutant measurement data pandas DataFrame used to train the model.
        x_features: The input dimensions used to train the model.
        y_features: The output dimensions used to train the model against.
        dataset: The dataset category to evaluate against.

    Returns:
        A tuple containing the NLPD and RMSE scores.
    """
    eval_df = df.loc[df["dataset"] == dataset]
    X = eval_df[x_features].values
    Y = eval_df[y_features].values
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)

    mean, var = GP.predict_y(X=t, R=R)
    nlpd = GP.negative_log_predictive_density(X=t, R=R, Y=Y)
    rmse = np.sqrt(np.nanmean((np.squeeze(Y) - np.squeeze(mean))**2))

    return nlpd, rmse