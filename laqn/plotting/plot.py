import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import timedelta
import bayesnewton

from typing import List, Tuple

SAVE_PATH = "plots/"
MAP_PATH = "data/London_Borough_Excluding_MHW.shp"

def set_size(width:float=433.0, fraction:float=1.0):
    """
    Sets figure dimensions to avoid scaling in LaTeX.

    Code from:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Args:
        width: Document textwidth or columnwidth in pts.
        fraction: Fraction of the width which you wish the figure to occupy.

    Returns:
        tuple holding the dimensions (width, height) of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

# Update default plt parameters
# https://stackoverflow.com/questions/17687213/how-to-obtain-the-same-font-style-size-etc-in-matplotlib-output-as-in-latex
fontsize = 22
plt.rcParams.update({
    "figure.figsize": (12, 9),              # 4:3 aspect ratio, can use set_size function defined above here
    "font.size" : fontsize,
    # "font.weight" : "bold",
    "axes.titlesize" : fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "font.family": "Helvetica",
    # "text.usetex": True,
    # "text.latex.preamble": (                # LaTeX preamble
    #    r"\usepackage[T1]{fontenc}"
    #     ... more packages if needed
    # )
})

def plot_site(model:bayesnewton.models.MarkovVariationalGP, df:pd.DataFrame, y_features:List[str], scalers:dict, code:str=None,
            site_lat:float=None, site_lon:float=None, title:str="Predictions after training", save:bool=False, save_path:str=SAVE_PATH) -> None:
    """
    Plots a time series of the predicted value vs. actual value at the given site code or coordinates.

    Args:
        model: A trained ST-SVGP model.
        df: The pandas DataFrame used to train the model.
        y_features: Variable(s) we are predicting against. Has only been tested for 1 feature at a time.
        code: Site code we are predicting at.
        site_lat: Latitude of the location we are predicting at.
        site_lon: Longitude of the location we are predicting at.
            User must provide either a site code or a pair of coordinates, but not both.
        title: Title of the plot.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    min_t, max_t = df["t"].min(), df["t"].max()
    t_plot = np.linspace(min_t, max_t, max_t-min_t+1).reshape(-1, 1)
    start_date, end_date = df["date"].min(), df["date"].max()
    delta = end_date - start_date
    dates = [start_date + timedelta(days=i) for i in range(delta.days+1)]
    
    if code:
        site_df = df.loc[df["code"] == code]
        site_lat, site_lon = site_df["latitude"].unique()[0], site_df["longitude"].unique()[0]
        scaled_lat = scalers["latitude"].transform(np.array([[site_lat]])).item()
        scaled_lon = scalers["longitude"].transform(np.array([[site_lon]])).item()
        save_code = code
    elif site_lat and site_lon:
        scaled_lat = scalers["latitude"].transform(np.array([[site_lat]])).item()
        scaled_lon = scalers["longitude"].transform(np.array([[site_lon]])).item()
        save_code = "unknown"
    else:
        print("Missing one of site code or site coordinates")
        return
    
    R_plot = np.tile(np.array([[scaled_lat, scaled_lon]]), [t_plot.shape[0], 1, 1])

    plt.figure()
    plt.subplots(facecolor="white", figsize=(18, 9))

    # PLOT ACTUAL VALUES
    Y_dates, Y_true = site_df["date"], site_df[y_features]
    plt.plot(Y_dates, Y_true, ".", label="Actual value", alpha=0.9, color="red")

    # PLOT PREDICTION WITH VARIANCE
    prediction_mean, prediction_var = model.predict_y(X=t_plot, R=R_plot)
    prediction_mean, prediction_var = prediction_mean.flatten(), prediction_var.flatten()
    (line,) = plt.plot(dates, prediction_mean, lw=1.5, label="Mean of predictive posterior")
    plt.fill_between(
        dates,
        prediction_mean-1.96*np.sqrt(prediction_var),
        prediction_mean+1.96*np.sqrt(prediction_var),
        color=line.get_color(),
        alpha=0.6,
        label=r"95% confidence interval"
    )

    # GET RMSE
    t_rmse = site_df["t"].values.reshape(-1, 1)
    R_rmse = np.tile(np.array([[scaled_lat, scaled_lon]]), [t_rmse.shape[0], 1, 1])
    rmse_mean, rmse_std = model.predict_y(X=t_rmse, R=R_rmse)
    rmse_mean, rmse_std = rmse_mean.flatten(), rmse_std.flatten()
    rmse = np.sqrt(np.nanmean((np.squeeze(Y_true) - np.squeeze(rmse_mean))**2))

    plt.title(f"{title}\nRMSE: {rmse}")
    plt.xlabel(f"Date")
    plt.ylabel(f"Concentration of NO$_{2}$ (µg/m$^{3}$)")
    
    plt.gcf().autofmt_xdate()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.legend(loc="upper right")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.savefig(fname=save_path + "prediction_" + save_code + ".png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_mutual_info(mutual_infos:List[np.array], labels:List[str], save=False, save_path=SAVE_PATH) -> None:
    """
    Plots the mutual information vs. the number of sensors in the network.

    Args:
        mutual_infos: List of mutual information numpy arrays.
        labels: List of labels for each mutual information array in mutual_infos.
            Length of labels must match length of mutual_infos.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    plt.figure()
    plt.subplots(facecolor="white", figsize=(12, 12))

    for ind, mi in enumerate(mutual_infos):
        plt.plot(list(range(1, len(mi)+1)), mi, label=labels[ind])
    
    plt.grid()
    plt.title("Mutual Information Gain by Sensor Added")
    plt.ylabel("Mutual Information")
    plt.xlabel("Number of Sensors")
    plt.xlim([0, 75])
    # plt.legend(loc="upper left")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save:
        plt.savefig(fname=save_path + "mutual_info.png", dpi=300, bbox_inches='tight')
    plt.show()

def gen_london_grid(grid_size:int=50) -> Tuple[np.array, np.array]:
    """
    Generates a grid over the area of London.

    Used when the contour values for the grid has already been previously generated and saved (see gen_contour_grid below).

    Args:
        grid_size: Size of the grid.

    Returns:
        The latitude and longitude meshgrids of the London grid.
    """
    lat_min, lat_max = 51.279, 51.7
    lon_min, lon_max = -0.533, 0.350
    lat_plot = np.linspace(lat_min, lat_max, num=grid_size).reshape(-1, 1)
    lon_plot = np.linspace(lon_min, lon_max, num=grid_size).reshape(-1, 1)
    LAT, LON = np.meshgrid(lat_plot, lon_plot)
    return LAT, LON

def gen_contour_grid(model:bayesnewton.models.MarkovVariationalGP, scalers:dict, t_max:int, t_min:int=0, grid_size:int=50) -> Tuple[np.array, np.array, np.array]:
    """
    Generates a grid over the area of London along with the corresponding variance (uncertainty) matrix.

    Args:
        model: A trained ST-SVGP model.
        scalers: Dict mapping the name of each scaled feature in model_df to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        t_max: Maximum time step to average the variance vector over.
        t_min: Minimum time step to average the variance vector over.
        grid_size: Size of the generated grid.

    Returns:
        The latitude and longitude meshgrids of the London grid and the variance (uncertainty) matrix in a tuple.
    """
    # MAKE LAT, LON GRID
    lat_min, lat_max = 51.279, 51.7
    lon_min, lon_max = -0.533, 0.350
    lat_plot = np.linspace(lat_min, lat_max, num=grid_size).reshape(-1, 1)
    lon_plot = np.linspace(lon_min, lon_max, num=grid_size).reshape(-1, 1)
    LAT, LON = np.meshgrid(lat_plot, lon_plot)

    # SCALE LAT, LON
    scaled_lat_plot = scalers["latitude"].transform(lat_plot).flatten()
    scaled_lon_plot = scalers["longitude"].transform(lon_plot).flatten()
    r1_plot, r2_plot = [], []
    for i in range(grid_size):
        for j in range(grid_size):
            r1_plot.append([scaled_lat_plot[i]])
            r2_plot.append([scaled_lon_plot[j]])
    r1_plot, r2_plot = np.array(r1_plot), np.array(r2_plot)

    # CALCULATE CONTOUR VALUES
    PRED_MEANS, PRED_VARS = [], []
    for t in range(t_min, t_max+1):
        t_plot = np.array([[t]])
        R_plot = np.tile(np.hstack((r1_plot, r2_plot)), [t_plot.shape[0], 1, 1])
        pred_mean, pred_var = model.predict_y(X=t_plot, R=R_plot)

    PRED_MEANS.append(pred_mean)
    PRED_VARS.append(pred_var)

    pred_mean_plot = np.mean(PRED_MEANS, axis=0).reshape(grid_size, grid_size).T
    pred_var_plot = np.mean(PRED_VARS, axis=0).reshape(grid_size, grid_size).T

    z_opt = model.kernel.z.value
    z_lat = scalers["latitude"].inverse_transform(z_opt[:, 0].reshape(-1, 1))
    z_lon = scalers["longitude"].inverse_transform(z_opt[:, 1].reshape(-1, 1))

    return LAT, LON, pred_var_plot

def plot_contour(model:bayesnewton.models.MarkovVariationalGP, df:pd.DataFrame, scalers:dict, t_max:int, pred_var_plot:np.array=None,
                t_min:int=0, title:int="Model Variance (Uncertainty) Across London", by_dataset:bool=True, map_path:str=MAP_PATH, save:bool=False, save_path:str=SAVE_PATH) -> np.array:
    """
    Plots a smooothed contour map of the uncertainty of the given ST-SVGP model over the area of London.

    Used in Experiment 1 analysis.

    Args:
        model: A trained ST-SVGP model.
        df: The pandas DataFrame used to train the model.
        scalers: Dict mapping the name of each scaled feature in model_df to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        t_max: Maximum time step to average the variance vector over.
        pred_var_plot: Variance vector to plot on the contour map.
            Same as the output of this function. It"s recommended to save this output as calculating the vector takes
            a long time to compute.
        t_min: Minimum time step to average the variance vector over.
        title: Title of the plot.
        by_dataset: If True, labels the sites by dataset category.
        map_path: Location of the SHP file containing London borough boundary information.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    if pred_var_plot is not None:
        LAT, LON = gen_london_grid()
    else:
        LAT, LON, pred_var_plot = gen_contour_grid(model, scalers, t_max, t_min=t_min)

    map_df = gpd.read_file(map_path)
    map_df = map_df.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.title(title)
    levels = np.linspace(6, 22, 9)

    plt.contourf(LON, LAT, pred_var_plot, cmap="coolwarm", levels=levels)
    map_df.plot(ax=ax, color="None")

    if by_dataset:
        groups = df.groupby("dataset")
        name_to_label = {"test": "Candidate", "train": "Training", "val": "Validation"}
        for name, group in groups:
            if name == "val":
                ax.scatter(group["longitude"], group["latitude"], marker="^", label=name_to_label[name], color="purple", edgecolors="black", s=50)
            elif name == "train":
                ax.scatter(group["longitude"], group["latitude"], marker="o", label=name_to_label[name], color="green", edgecolors="black", s=50)
            elif name == "test":
                ax.scatter(group["longitude"], group["latitude"], marker="s", label=name_to_label[name], color="orange", edgecolors="black", s=50)
    else:
        ax.scatter(df["longitude"], df["latitude"], marker="o", label="NO$_2$ Sensor Site", color="green", edgecolors='black', s=50)

    # plt.scatter(z_lon, z_lat, label="inducing points")#, color="purple")
    # plt.legend(title="Site Category")
    plt.legend(title="Site Category", loc='center left', bbox_to_anchor=(1.2, 0.5))
    plt.colorbar(label="Model Prediction Variance")

    # plt.annotate("RB4", (0.030858+0.01, 51.57661), fontsize=12)
    # plt.annotate("EA8", (-0.265617-0.035, 51.518948), fontsize=12)
    # plt.annotate("c*", (0.092, 51.433), fontsize=12)

    plt.ylabel("Latitude (WGS84)")
    plt.xlabel("Longitude (WGS84)")

    if save:
        plt.savefig(fname=save_path + "contour_plot.png", dpi=300)
    plt.show()

    return pred_var_plot

def gen_contour_diff_df(df_1:pd.DataFrame, df_2:pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new pandas DataFrame which categorizes the difference between the two DataFrames' train, test, and validation
    sites into {"Kept", "Validation", "Dropped", "Added"}.

    Input DataFrames are the DataFrames that were used to train two separate ST-SVGP models. Used in Experiment 1 analysis.
    
    Args:
        df_1: DataFrame used to train the first ST-SVGP model.
        df_2: DataFrame used to train the second (or some subsequent) ST-SVGP model.
    
    Returns:
        A copy of the second DataFrame with the added column "Category" which has values {"Kept", "Validation", "Dropped", "Added"}.
    """

    train_sites = set(df_1.loc[df_1["dataset"] == "train"]["code"].unique())
    new_model_sites = set(df_2.loc[df_2["dataset"] == "train"]["code"].unique())
    added_sites = new_model_sites - train_sites
    dropped_sites = train_sites - new_model_sites
    kept_sites = train_sites.intersection(new_model_sites)
    val_sites = set(df_1.loc[df_1["dataset"] == "val"]["code"].unique())

    site_map = {}
    site_map.update({kept_site: "Kept" for kept_site in kept_sites})
    site_map.update({val_site: "Validation" for val_site in val_sites})
    site_map.update({dropped_site: "Dropped" for dropped_site in dropped_sites})
    site_map.update({added_site: "Added" for added_site in added_sites})

    df_2["Category"] = df_2["code"].map(site_map)

    return df_2.copy()

def plot_contour_diff(diff_df:pd.DataFrame, pred_var_plot_diff:np.array, title:str="Difference in Model Prediction Variance (Optimized-Original Model Uncertainty) Across London", map_path:str=MAP_PATH, save:bool=False, save_path:str=SAVE_PATH) -> None:
    """
    Plots a smooothed contour map of the difference in uncertainty between two ST-SVGP models.

    Used in Experiment 1 analysis.
    
    Args:
        diff_df: DataFrame holding the difference in uncertainty between two models.
            Same form as the output of gen_contour_diff_df above.
        pred_var_plot_diff: Difference in variance (uncertainty) between two models.
        title: Title of the plot.
        map_path: Location of the SHP file containing London borough boundary information.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    LAT, LON = gen_london_grid()

    map_df = gpd.read_file(map_path)
    map_df = map_df.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(20,20))

    plt.title(title)
    plt.contourf(LON, LAT, pred_var_plot_diff, cmap="coolwarm")#levels=levels, cmap="Blues")
    map_df.plot(ax=ax, color="None")

    groups = diff_df.groupby("Category")
    for name, group in groups:
        if name == "Validation":
            ax.scatter(group["longitude"], group["latitude"], marker="^", label=name, color="purple", edgecolors="black", s=50)
        elif name == "Kept":
            ax.scatter(group["longitude"], group["latitude"], marker="o", label=name, color="green", edgecolors="black", s=50)
        elif name == "Added":
            ax.scatter(group["longitude"], group["latitude"], marker="o", label=name, color="orange", edgecolors="black", s=50)
        elif name == "Dropped":
            ax.scatter(group["longitude"], group["latitude"], marker="X", label=name, color="red", edgecolors="black", s=50)

    # plt.scatter(z_lon, z_lat, label="inducing points")#, color="purple")
    # plt.legend(title="Site Category")
    plt.legend(title="Site Category", loc='center left', bbox_to_anchor=(1.23, 0.5))

    plt.colorbar(label="Difference in Model Prediction Variance (Optimized-Original)")

    # plt.annotate("RB4", (0.030858+0.01, 51.57661), fontsize=12)
    # plt.annotate("EA8", (-0.265617-0.035, 51.518948), fontsize=12)
    # plt.annotate("c*", (0.092, 51.433), fontsize=12)

    plt.ylabel("Latitude (WGS84)")
    plt.xlabel("Longitude (WGS84)")

    if save:
        plt.savefig(fname=save_path + "contour_diff_plot.png", dpi=300)
    plt.show()

def plot_final_rec(krause_df:pd.DataFrame, title:str, map_path:str=MAP_PATH, save:bool=False, save_path:str=SAVE_PATH) -> None:
    """
    Plots the new vs. existing sites across the area of London.
    
    Used in Experiment 2 analysis.

    Args:
        krause_df: pandas DataFrame of the same form as the output of krause_alg (see optimization module)
        title: Title of the plot.
        map_path: Location of the SHP file containing London borough boundary information.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    map_df = gpd.read_file(map_path)
    map_df = map_df.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(20,20))

    plt.title(title)

    map_df.plot(ax=ax, color="w", edgecolors="black")
    rec_krause_df = krause_df.loc[krause_df["order"] <= 75] # Can change this number
    rec_krause_df["Category"] = rec_krause_df["code"].apply(lambda row: "New" if row.startswith("U") else "Existing")

    groups = rec_krause_df.groupby("Category")
    for name, group in groups:
        if name == "New":
            ax.scatter(group["longitude"], group["latitude"], label=name, color="orange", edgecolors="black", s=50)
        elif name == "Existing":
            ax.scatter(group["longitude"], group["latitude"], label=name, color="green", edgecolors="black", s=50)

    plt.legend(title="Site Category")

    plt.ylabel("Latitude (WGS84)")
    plt.xlabel("Longitude (WGS84)")

    if save:
        plt.savefig(fname=save_path + "final_rec_plot.png", dpi=300)
    plt.show()

def plot_covariance(model, df, scalers, code):
    """
    TODO: Plots a heatmap of site covariance across the area of London.
    """
    return