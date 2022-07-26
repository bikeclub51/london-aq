import pandas as pd
import math
import json
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd

from laqn.plotting import plot # Used to import default pyplot style changes

from typing import Tuple

def generate_placement_sets(model_df:pd.DataFrame, scalers:dict, site_coords_path:str="data/monitoring_sites.csv", 
                            map_path:str="data/London_Borough_Excluding_MHW.shp", london_boroughs_path:str="data/london_boroughs.json",
                            n:int=1000, plot:bool=False, include_test:bool=True, include_val:bool=False, save:bool=False, save_path:str="plots/") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates set S of existing sensor locations and set U of locations which we are interested in but cannot place a sensor at.

    Args:
        model_df: LAQN pollutant measurement data pandas DataFrame.
        scalers: Dict mapping the name of each scaled feature in model_df to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        site_coords_path: Path to a CSV file containing the site codes and their coordinates (latitude, longitude). This file
            can be found in the Dropbox data folder linked in the Data Download module or downloaded directly via the LAQN API:
            https://api.erg.ic.ac.uk/AirQuality/help/operations/GetMonitoringSitesJson.
        map_path: Path to a SHP file of the London boroughs used for plotting (see plot below).
        london_boroughs_path: Path to a JSON file containing the London boroughs' geometry as Polygons. This file can be found in 
            the Dropbox data folder linked in the Data Download module or downloaded directly here: https://skgrange.github.io/data.html.
        n: Size of the grid used to generate set U. For example, n=1000 will generate a grid of 1000**(0.5) x 1000**(0.5)
            across the area of London and keep the points (site locations) which fall within the London boundaries.
        plot: If True, will plot sets S and U on a map of London. Requires a SHP file of the London boroughs.
            This file can be found in the Dropbox data folder linked in the Data Download module or downloaded directly
            here: https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london. We use London_Borough_Excluding_MHW.shp 
            found in statistical-gis-boundaries-london.zip.
        include_test: If True, includes the test sites in set S.
        include_val: If True, includes the validation sites in set S.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.

    Returns:
        A (pandas DataFrame, pandas DataFrame) tuple of sets S and U with columns 
        ["code", "latitude", "longitude", "scaled_latitude", "scaled_longitude"].
    """
    S_df = generate_set_S(model_df, scalers, site_coords_path, map_path, plot, include_test, include_val, save, save_path)
    U_df = generate_set_U(scalers, london_boroughs_path, n, map_path, plot, save, save_path)
    return S_df, U_df


def generate_set_S(model_df:pd.DataFrame, scalers:dict, site_coords_path:str, map_path:str,
                    plot:bool, include_test:bool, include_val:bool, save:bool, save_path:str) -> pd.DataFrame:
    """
    Generates set S of existing sensor locations.

    Args:
        model_df: LAQN pollutant measurement data pandas DataFrame.
        scalers: Dict mapping the name of each scaled feature in model_df to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        site_coords_path: Path to a CSV file containing the site codes and their coordinates (latitude, longitude). This file
            can be found in the Dropbox data folder linked in the Data Download module or downloaded directly via the LAQN API:
            https://api.erg.ic.ac.uk/AirQuality/help/operations/GetMonitoringSitesJson.
        map_path: Path to a SHP file of the London boroughs used for plotting (see plot below).
        plot: If True, will plot set S on a map of London. Requires a SHP file of the London boroughs.
            This file can be found in the Dropbox data folder linked in the Data Download module or downloaded directly
            here: https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london. We use London_Borough_Excluding_MHW.shp 
            found in statistical-gis-boundaries-london.zip.
        include_test: If True, includes the test sites.
        include_val: If True, includes the validation sites.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.

    Returns:
        A pandas DataFrame representing set S with columns ["code", "latitude", "longitude", "scaled_latitude", "scaled_longitude"].
    """
    rename_cols = {"SiteCode": "code", "Latitude": "latitude", "Longitude": "longitude"}
    site_df = pd.read_csv(site_coords_path).rename(columns=rename_cols)
    train_sites = set(model_df.loc[model_df["dataset"] == "train"]["code"].unique())
    S_df = site_df.loc[site_df["code"].isin(train_sites)]
    if include_test:
        test_sites = set(model_df.loc[model_df["dataset"] == "test"]["code"].unique())
        new_sites = test_sites - train_sites
        S_df = S_df.append(site_df.loc[site_df["code"].isin(new_sites)])
    if include_val:
        val_sites = set(model_df.loc[model_df["dataset"] == "val"]["code"].unique())
        new_sites = val_sites - train_sites
        S_df = S_df.append(site_df.loc[site_df["code"].isin(new_sites)])
    S_df["scaled_latitude"] = scalers["latitude"].transform(S_df["latitude"].values.reshape(-1, 1))
    S_df["scaled_longitude"] = scalers["longitude"].transform(S_df["longitude"].values.reshape(-1, 1))
    
    if plot:
        fig, ax = plt.subplots(figsize=(20, 20))

        map_df = gpd.read_file(map_path)
        map_df = map_df.to_crs("EPSG:4326")
        map_df.plot(ax=ax, color="None")

        ax.scatter(S_df["longitude"], S_df["latitude"], label='NO$_2$ Sensor Site', marker='o', color="green", edgecolors='black', s=50)
        plt.title("LAQN NO$_2$ Sensor Site Locations")
        plt.ylabel('Latitude (WGS84)')
        plt.xlabel('Longitude (WGS84)')
        plt.legend()

        if save:
            plt.savefig(fname=save_path + "set_S.png", dpi=300)

        plt.show()

    return S_df


def generate_set_U(scalers:dict, london_boroughs_path:str, n:int, map_path:str, plot:bool, save:bool, save_path:str) -> pd.DataFrame:
    """
    Generates set S of existing sensor locations and set U of locations which we are interested in but cannot place a sensor at.

    Args:
        scalers: Dict mapping the name of each scaled feature to its respective StandardScaler object.
            Must have features "latitude" and "longitude".
        london_boroughs_path: Path to a JSON file containing the London boroughs' geometry as Polygons. This file can be found in 
            the Dropbox data folder linked in the Data Download module or downloaded directly here: https://skgrange.github.io/data.html.
        n: Size of the grid used to generate set U. For example, n=1000 will generate a grid of 1000**(0.5) x 1000**(0.5)
            across the area of London and keep the points (site locations) which fall within the London boundaries.
        map_path: Path to a SHP file of the London boroughs used for plotting (see plot below).
        plot: If True, will plot set U on a map of London. Requires a SHP file of the London boroughs.
            This file can be found in the Dropbox data folder linked in the Data Download module or downloaded directly
            here: https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london. We use London_Borough_Excluding_MHW.shp 
            found in statistical-gis-boundaries-london.zip.
        save: If True, saves the plot to the save_path (see below).
        save_path: Path to figures folder.

    Returns:
        A pandas DataFrame representing set U with columns ["code", "latitude", "longitude", "scaled_latitude", "scaled_longitude"].
    """
    # Retrieve London coordinate boundaries
    london_burough_boundaries_df = get_london_boundaries(london_boroughs_path)
    # london_gdf = gpd.GeoDataFrame(
    #     london_burough_boundaries_df, geometry=gpd.points_from_xy(london_burough_boundaries_df.Longitude, london_burough_boundaries_df.Latitude))
    
    # Find max and min latitude and longitude coordinates, extremes
    min_latitude = math.floor(london_burough_boundaries_df.Latitude.min())
    max_latitude = math.ceil(london_burough_boundaries_df.Latitude.max())
    
    min_longitude = math.floor(london_burough_boundaries_df.Longitude.min())
    max_longitude = math.ceil(london_burough_boundaries_df.Longitude.max())
    
    # Produce equally distributed values of latitude and longitude coordinates within London max and min boundaries
    latitudes = np.linspace(min_latitude, max_latitude, int(n**0.5))
    longitudes =  np.linspace(min_longitude, max_longitude, int(n**0.5))
    u_sites = {'code': [], 'latitude': [], 'longitude': []}
    
    # Creates map_df that contains shapely polygons for each burough
    map_df = gpd.read_file(map_path).to_crs("EPSG:4326")
    map_df = map_df.explode(index_parts=False)

    # Creates U locations with corresponding code
    grid_len = int(n**0.5)
    for i in range(grid_len):
        for j in range(grid_len):
            site_code = f"U_{i*grid_len + j}" # U location code
            num_decimal_places = 13
            float_long = round(float(longitudes[i]), num_decimal_places)
            float_lat = round(float(latitudes[j]), num_decimal_places)
            
            # Creates Point instance of coordinate and a bounding polygon that contains the point
            coord = Point(float_long, float_lat)
            p_range = [-0.00001, 0.0, 0.00001]
            poly_point = Polygon([(coord.x + lam_lat, coord.y + lam_lon) for lam_lat in p_range for lam_lon in p_range])
            
            # Finds whether the bounding coordinate polygon intersects with the burough boundaries
            for burough in map_df.geometry:
                if burough.intersects(poly_point):                   
                    u_sites['code'].append(site_code)
                    u_sites['latitude'].append(latitudes[j])
                    u_sites['longitude'].append(longitudes[i])

    U_df = pd.DataFrame(u_sites)
    U_df["scaled_latitude"] = scalers["latitude"].transform(U_df["latitude"].values.reshape(-1, 1))
    U_df["scaled_longitude"] = scalers["longitude"].transform(U_df["longitude"].values.reshape(-1, 1))

    U_df["code"] = [f"U_{i}" for i in range(U_df.shape[0])]

    if plot:
        fig, ax = plt.subplots(figsize=(20, 20))

        map_df.plot(ax=ax, color="None")

        ax.scatter(U_df["longitude"], U_df["latitude"], label='Location of Interest', marker='o', color="blue", edgecolors='black', s=50)
        plt.title("Evenly Distributed Locations Across London (Set L)")
        plt.ylabel('Latitude (WGS84)')
        plt.xlabel('Longitude (WGS84)')
        plt.legend()

        if save:
            plt.savefig(fname=save_path + "set_U.png", dpi=300)
    
        plt.show()

    return U_df

def get_london_boundaries(london_boroughs_path:str) -> pd.DataFrame:
    """
    Reads the London boroughs boundaries JSON file and converts it into a pandas DataFrame.

    Args:
        london_boroughs_path: JSON file of London boroughs' geometries.

    Returns:
        A pandas DataFrame holding the boundary information found in the JSON file.
    """

    boundaries = []

    london_burough_boundaries = json.load(open(london_boroughs_path))
    for burough in london_burough_boundaries["features"]:
        for coordinate in burough["geometry"]["coordinates"][0][0]:
            boundaries.append(coordinate)
    
    london_burough_boundaries_df = pd.DataFrame(boundaries, columns=['Longitude', 'Latitude'])
    london_burough_boundaries_df = london_burough_boundaries_df[london_burough_boundaries_df.columns[::-1]]

    return london_burough_boundaries_df