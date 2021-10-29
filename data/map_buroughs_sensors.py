import pandas as pd
import requests
import json
import codecs
from copy import deepcopy
import pandas as pd
import plotly
import numpy as np
from matplotlib import pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

## SET GLOBAL VARIABLES

london_lat, london_lon = 51.5074, -0.1278
species = ["NO2", "CO", "O3", "SO2", "PM10", "PM25"]
ghost_data = {'Latitude': [london_lat]*6, 'Longitude': [london_lon]*6, "SpeciesCode": species, "ActiveYear": [1987]*6}
ghost_df = pd.DataFrame.from_dict(ghost_data)

LAT_NOISE = 0.001
LON_NOISE = 0.002

lat_noise = {"NO2": LAT_NOISE, "CO": -LAT_NOISE, "O3": 0, "SO2": 0, "PM10": LAT_NOISE, "PM25": -LAT_NOISE}
lon_noise = {"NO2": 0, "CO": 0, "O3": LON_NOISE, "SO2": -LON_NOISE, "PM10": LON_NOISE, "PM25": -LON_NOISE}

def get_air_quality_data(group_name):
  '''
  API GET request to London Open Air Quality API
  '''
  url = f"http://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName={group_name}/Json"

  resp = requests.get(url)
  decoded_data = resp.content.decode('utf-8-sig')
  json_data = json.loads(decoded_data)
  
  return json_data


def cross_join(left, right):
    new_rows = [] if right else left
    for left_row in left:
        for right_row in right:
            temp_row = deepcopy(left_row)
            for key, value in right_row.items():
                temp_row[key] = value
            new_rows.append(deepcopy(temp_row))
    return new_rows


def flatten_list(data):
    for elem in data:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


def json_to_dataframe(data_in):
    '''
    Helper functions used to parse nested JSON file into a pandas DataFrame.
    Code adapted from: https://stackoverflow.com/a/63500540
    '''
    def flatten_json(data, prev_heading=''):
        if isinstance(data, dict):
            rows = [{}]
            for key, value in data.items():
                rows = cross_join(rows, flatten_json(value, key))
        elif isinstance(data, list):
            rows = []
            for i in range(len(data)):
                [rows.append(elem) for elem in flatten_list(flatten_json(data[i], prev_heading))]
        else:
            rows = [{prev_heading[1:]: data}]
        return rows

    return pd.DataFrame(flatten_json(data_in))


def get_yearly_range(sdate, edate):
  years = list(range(sdate.year, edate.year+1))
  return years


def add_lat_noise(row):
  return row['Latitude'] + lat_noise[row['SpeciesCode']]


def add_lon_noise(row):
  return row['Longitude'] + lon_noise[row['SpeciesCode']]


def plot_pollutant_site_types(yearly_df, pollutant):
    
    pollutant_df = yearly_df.loc[yearly_df["SpeciesCode"] == pollutant]

    site_types = ['Suburban', 'Kerbside', 'Urban Background', 'Industrial', 'Roadside', 'Rural']
    site_order = {"NO2": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'],
                    "CO": ['Urban Background', 'Roadside', 'Suburban', 'Kerbside', 'Industrial', 'Rural'],
                    "O3": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'], 
                    "SO2": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'], 
                    "PM10": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'],
                    "PM25": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural']}
    start_year = {"NO2": 1987,
                    "CO": 1987,
                    "O3": 1990,
                    "SO2": 1990,
                    "PM10": 1993,
                    "PM25": 2001}

    n = len(list(range(start_year[pollutant], 2022)))
    ghost_data = {'Latitude': [london_lat]*6*n, 'Longitude': [london_lon]*6*n, "SiteType": site_types*n, "ActiveYear": list(range(start_year[pollutant], 2022))*6}
    ghost_df = pd.DataFrame.from_dict(ghost_data)
    plot_df = pd.concat([ghost_df, pollutant_df])

    fig = px.scatter_mapbox(plot_df, lat='Latitude', lon='Longitude', color="SiteType", animation_frame="ActiveYear", animation_group="ActiveYear",
        width=1500, 
        height=750,
        zoom=10)

    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    fig.update_layout(mapbox_style="carto-positron", margin = dict(l = 0, r = 0, t = 10, b = 0))

    fig.add_trace(go.Scattermapbox(
            lat=[london_lat],
            lon=[london_lon],
            text='London',
            mode='markers',
            showlegend=False,
            marker=go.scattermapbox.Marker(
                size=8,
                color='white',
                opacity=1,
            ),
        ))

    # fig.show()
    return fig


def plot_all_pollutants_year(yearly_df, year):
    year_df = yearly_df.loc[yearly_df["ActiveYear"] == year]

    site_types = ['Suburban', 'Kerbside', 'Urban Background', 'Industrial', 'Roadside', 'Rural']
    site_order = {"NO2": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'],
                    "CO": ['Urban Background', 'Roadside', 'Suburban', 'Kerbside', 'Industrial', 'Rural'],
                    "O3": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'], 
                    "SO2": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'], 
                    "PM10": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural'],
                    "PM25": ['Urban Background', 'Roadside', 'Kerbside', 'Suburban', 'Industrial', 'Rural']}
    start_year = {"NO2": 1987,
                    "CO": 1987,
                    "O3": 1990,
                    "SO2": 1990,
                    "PM10": 1993,
                    "PM25": 2001}

    # ghost_data = {'Latitude': [london_lat]*6, 'Longitude': [london_lon]*6, "SiteType": site_types*n, "ActiveYear": list(range(start_year[pollutant], 2022))*6}
    # ghost_df = pd.DataFrame.from_dict(ghost_data)
    # plot_df = pd.concat([ghost_df, pollutant_df])

    fig = px.scatter_mapbox(year_df, lat='Latitude', lon='Longitude', color="SpeciesCode", hover_name="SiteName",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    fig.update_layout(mapbox_style="carto-positron", margin = dict(l = 0, r = 0, t = 10, b = 0))

    fig.add_trace(go.Scattermapbox(
            lat=[london_lat],
            lon=[london_lon],
            text='London',
            mode='markers',
            showlegend=False,
            marker=go.scattermapbox.Marker(
                size=8,
                color='white',
                opacity=1,
            ),
        ))

    fig.show()
    return fig

def get_local_authorities_geojson(file):
        '''
        Open geojson file storing local authority data
        '''
        local_auth_json = json.load(open(file))
        return local_auth_json


def get_local_authorities_df(local_auth_json):
    '''
    Gets local authority ids and names from geojson file. Converts data into a DataFrame
    '''

    local_auth_dict = {"LAD13CD": [], "LAD13CDO": [], "AuthorityName": [], "Color": []}
    columns = list(local_auth_dict.keys())
    i = 0
    for local_auth in local_auth_json["features"]:
        for col in columns[:-2]:
            # print('dict', local_auth_dict[col])
            # print('geojson', local_auth[col])
            local_auth_dict[col].append(local_auth["properties"][col])
        local_auth_dict["AuthorityName"].append(local_auth["properties"]["LAD13NM"])
        local_auth_dict["Color"].append(i)
        i += 1
    
    df = pd.DataFrame(local_auth_dict)
    return df


def map_buroughs_census_data(pollutant, map_pollutant, population=False, migration=False, health=False):
    local_auth_json = get_local_authorities_geojson("data/gb_local_authorities.geojson")
    local_auth_df = get_local_authorities_df(local_auth_json)
    # print("Local Authorities")
    # print(local_auth_df)

    census_data = pd.DataFrame()
    if population:
        census_data = pd.read_excel("data/london-unrounded-data.xls", sheet_name="Persons")
        locations_col = "LAD13CD"
        value_col = "TotalPopulation"
        geojson_feature_id = "properties.LAD13CD"
        title = "Population Data"
        color_range = ["blue", "green"]
    
    elif migration:
        census_data =  pd.read_excel()
        locations_col = "LAD13CD"
        value_col = "Borough"
        title = "Migrant Population Data"

    elif health:
        census_data =  pd.read_excel("data/visualisation-data-health/HEALTH.xlsx", sheet_name="2011")
        locations_col = "Borough"
        value_col = "Very good health"
        geojson_feature_id = "properties.LAD13NM"
        title = "Population Health Levels Data"
        color_range = ["red", "yellow", "green"]

    # print("Census Data")
    # print(census_data)

    fig = px.choropleth_mapbox(data_frame=census_data, geojson=local_auth_json, locations= locations_col, 
                                featureidkey=geojson_feature_id, color=value_col, mapbox_style="carto-positron", zoom=9, 
                                center = {"lat": london_lat, "lon": london_lon}, color_continuous_scale= color_range,  
                                opacity=0.4)

    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    fig.add_trace(map_pollutant.data[0])
    # print(fig.data)
    # for i in enumerate(map_pollutant.data):
    #     # print(i, frame)
    #     fig.data += (map_pollutant.data[i],)

    fig.update_layout(height=800,
        title_text= pollutant + ' Sensors & ' + title + ' (LAQN)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            )
        )
    fig.update_layout(margin={"r":1,"t":80,"l":1,"b":1})
    # fig.show()
    return fig


## GETS DATA FROM LAQN API

data = get_air_quality_data("London")
df = json_to_dataframe(data)

df[["Latitude", "Longitude"]] = df[["Latitude", "Longitude"]].apply(pd.to_numeric)
df[["LatitudeWGS84", "LongitudeWGS84"]] = df[["LatitudeWGS84", "LongitudeWGS84"]].apply(pd.to_numeric)
df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']] = df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']].apply(pd.to_datetime)
df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']] = df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']].fillna(date.today())

## GETS YEARLY DF

yearly_df = df.copy()
yearly_df['ActiveYear'] = [get_yearly_range(s.date(), e.date()) for s, e in zip(pd.to_datetime(yearly_df['DateMeasurementStarted']), pd.to_datetime(yearly_df['DateMeasurementFinished']))]
yearly_df = yearly_df.explode('ActiveYear')
yearly_df = yearly_df.dropna(subset=["ActiveYear"])
yearly_df = yearly_df.sort_values(by="ActiveYear")

yearly_df['JitterLatitude'] = yearly_df.apply(lambda row: add_lat_noise(row), axis=1)
yearly_df['JitterLongitude'] = yearly_df.apply(lambda row: add_lon_noise(row), axis=1)

map_pollutant = plot_pollutant_site_types(yearly_df, pollutant="CO")

map_data_2020 = plot_all_pollutants_year(yearly_df, 2020)

map_burough = map_buroughs_census_data("2020", map_data_2020, health=True)

map_burough.show()