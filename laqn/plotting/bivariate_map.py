"""
Plot a bivariate map of ST-SVGP model uncertainty vs. census data per London borough.

Census data for 2011 can be gathered from:
https://datashine.org.uk/#table=QS201EW&col=QS201EW0002&ramp=YlOrRd&layers=BTTT&zoom=11&lon=-0.1577&lat=51.4997
"""
import bayesnewton
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from shapely.geometry import Polygon, Point
import geopandas as gpd

from typing import Tuple

def get_census_data(file_path:str, map_path:str="data/London_Ward_CityMerged.shp") -> pd.DataFrame:
    """
    Loads census data into pandas DataFrame.

    Args:
        file_path: Path to the census data CSV file.
        map_path: Location of the SHP file containing London ward boundary information.

    Returns:
        A pandas DataFrame of the census data CSV file.
    """

    # attribute_fp = {"race": "blackAfricanCaribbean", 
    #     "poverty": "lowEconomicActivity", 
    #     "wealth": "census_economic_activity",
    #     "health": "longTermHealthConditionDisability_limitedDayToDayALot",
    #     "population": "populationDensity"}

    census_df = pd.read_csv(file_path, index_col=False)
    census_df = census_df.groupby(["Ward Name", "LA Name"], as_index=False).mean()
    census_df = census_df.rename(columns={"Ward Name": "NAME", "LA Name": "BOROUGH"})

    map_df = gpd.read_file(map_path).to_crs("EPSG:4326")
    
    census_map_df = map_df.merge(census_df, on=["NAME", "BOROUGH"])

    return census_map_df

    
def sample_wards(ward_df:pd.DataFrame, scalers:dict, n:int=5) -> pd.DataFrame:
    """
    Samples random locations (latitude and longitude coordinates) from the wards in the given pandas DataFrame.

    Args:
        ward_df: pandas DataFrame containing London borough boundary geometries.
            Same form as the output of get_census_data above.
        scalers: Dict mapping the name of each scaled feature in model_df to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        n: Number of random samples to take from each ward/borough.

    Returns:
        A copy of the input DataFrame with new columns ["samples", "scaled_samples"] which hold numpy arrays of the
        randomly sampled locations.
    """
    def sample_polygon(polygon) -> Tuple[np.array, np.array]:
        """
        Samples n random locations within the given polygon.

        Code adapted from: https://gis.stackexchange.com/a/294403

        Args:
            polygon: Polygon to sample from.

        Returns:
            A tuple of the locations and scaled locations, each in a numpy array.
        """
        points, scaled_points = [], []
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        count = 0
        while count < n:
            lat, lon = random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon)
            point = Point(lon, lat)
            p_range = [-0.000001, 0.0, 0.000001]
            poly_point = Polygon([(point.x + lam_lon, point.y + lam_lat) for lam_lon in p_range for lam_lat in p_range])
            if polygon.intersects(poly_point):
                points.append([lat, lon])
                scaled_lat = scalers['latitude'].transform(np.array([[lat]])).item()
                scaled_lon = scalers['longitude'].transform(np.array([[lon]])).item()
                scaled_points.append([scaled_lat, scaled_lon])
                count += 1
        return np.array(points), np.array(scaled_points)
    
    wards = ward_df['geometry'].values
    samples, scaled_samples = [], []

    for ward in wards:
        sample, scaled_sample = sample_polygon(ward)
        samples.append(sample)
        scaled_samples.append(scaled_sample)
    
    new_df = ward_df.copy()
    new_df["samples"] = samples
    new_df["scaled_samples"] = scaled_samples
    return new_df


def get_ward_uncertainty(model:bayesnewton.models.MarkovVariationalGP, ward_df:pd.DataFrame, max_t:int=0, min_t:int=0) -> pd.DataFrame:
    """
    Calculates the model uncertainty within each sample location of each ward.

    Args:
        model: A trained ST-SVGP model.
        ward_df: pandas DataFrame containing London borough boundary geometries.
            Same form as the output of get_census_data above.
        max_t: Maximum time step to average the model uncertainty over.
        min_t: Minimum time step to average the model uncertainty over.

    Returns:
        A copy of the given pandas DataFrame with a new column "uncertainty" which holds the uncertainty calculated for each sample
        within each ward.
    """
    samples = ward_df["scaled_samples"].values
    
    t_ward = np.linspace(min_t, max_t, max_t-min_t+1).reshape(-1, 1)
    uncertainties = []
    count = 0
    for sample_set in samples:
        R_ward = np.tile(sample_set, [t_ward.shape[0], 1, 1])
        ward_mean, ward_var = model.predict_y(X=t_ward, R=R_ward)
        mean_var = np.mean(ward_var)
        uncertainties.append(mean_var.item())
        print(count, mean_var)
        count += 1

    new_df = ward_df.copy()
    new_df["uncertainty"] = uncertainties
    return new_df

def color_from_bivariate_data(Z1:np.array, Z2:np.array, cmap1:plt.cm=plt.cm.Blues, cmap2:plt.cm=plt.cm.Reds) -> np.array:
    """
    Generates a color value for a bivariate map given two arrays of values.

    Code adapted from: https://gist.github.com/wolfiex/64d2faa495f8f0e1b1a68cdbdf3817f1#file-bivariate-py

    Args:
        Z1: Array of values for the first variable of interest.
        Z2: Array of values for the second variable of interest.
        cmap1: Colormap for the first variable of interest.
        cmap2: Colorap for the second variable of interest.

    Returns:
        A numpy array holding the bivariate color value for the given data.
    """
    z1mn = Z1.min()
    z2mn = Z2.min()
    z1mx = Z1.max()
    z2mx = Z2.max()        

    # Rescale values to fit into colormap range (0->255)
    Z1_plot = np.array(255*(Z1-z1mn)/(z1mx-z1mn), dtype=np.int)
    Z2_plot = np.array(255*(Z2-z2mn)/(z2mx-z2mn), dtype=np.int)

    Z1_color = cmap1(Z1_plot)
    Z2_color = cmap2(Z2_plot)
    
    # Color for each point
    Z_color = np.sum([Z1_color, Z2_color], axis=0)/2.0

    return Z_color

def bivariate_plot(plot_df:pd.DataFrame, var_1:str, var_2:str, label_1:str=None, label_2:str=None,
                title:str="London Wards", to_bin:bool=False, save:bool=False, save_path:str="plots/") -> None:

    """
    Plots a bivariate heatmap of the two given variables across the area of London by borough.

    Code adapted from: https://gist.github.com/wolfiex/64d2faa495f8f0e1b1a68cdbdf3817f1#file-bivariate-py

    Args:
        plot_df: The pandas DataFrame used to plot. Must have London borough geometries and columns for the given
            variables.
        var_1: First variable to plot against. Must be a column in plot_df.
        var_2: Second variable to plot against. Must be a column in plot_df.
        label_1: Axis label for the first variable.
        label_2: Axis label for the second variable.
        title: Title for the plot.
        to_bin: If True, bins each value into one of the nine colors on the heatmap. Otherwise, uses the raw color value.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15,15), gridspec_kw={"width_ratios": [3, 1]})
    ax1, ax2 = axs[0], axs[1]

    num_bins = 3

    def bin_var(var):
        """
        Bins the given variable.

        Args:
            var: Variable to bin.

        Returns:
            A copy of the plot_df with a new f"binned_{var}" column containing the binned value.
        """
        labels = list(range(num_bins))
        plot_df[f"binned_{var}"] = pd.cut(plot_df[var], 3, labels=labels)
        plot_df[f"binned_{var}"] = plot_df[f"binned_{var}"].astype(np.int8)
        return plot_df.copy()

    if to_bin:
        z1, z2 = bin_var(var_1)[f"binned_{var_1}"].values, bin_var(var_2)[f"binned_{var_2}"].values
    else:
        z1, z2 = plot_df[var_1].values, plot_df[var_2].values

    C_map = color_from_bivariate_data(z1, z2)

    ax1.set_facecolor('palegreen')
    ax1.set_title(title)
    ax1.set_ylabel('Latitude (WGS84)')
    ax1.set_xlabel('Longitude (WGS84)')
    plot_df.plot(ax=ax1, color=C_map, edgecolor="black")

    xx, yy = np.mgrid[0:num_bins,0:num_bins]
    C_map = color_from_bivariate_data(xx,yy)

    ax2.imshow(C_map)
    # ax2.set_title('Bivariate Color Map')
    if not label_1:
        label_1 = var_1
    if not label_2:
        label_2 = var_2
    ax2.set_xlabel(f"{label_2} (Low to High)")
    ax2.set_ylabel(f"{label_1} (Low to High)")
    ax2.set_ylim((-0.5,0.5+(yy.max()-yy.min())))     

    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticks([])

    fig.tight_layout()
    if save:
        fig.savefig(save_path+"bivariate_plot.png", facecolor="white", edgecolor="none", dpi=300)
    plt.show()