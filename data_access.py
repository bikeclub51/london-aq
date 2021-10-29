# installing modules
import requests
import json
import codecs
from copy import deepcopy
import pandas as pd
import plotly
import numpy as np
from matplotlib import pyplot as plt
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import openpyxl

def get_air_quality_data(group_name):
    '''
    API GET request to London Open Air Quality API
    '''
    # url = "http://api.erg.ic.ac.uk/AirQuality/Annual/MonitoringObjective/GroupName=" \
    #     + group_name + "/Json"

    url = 'http://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName=' + group_name + '/Json'
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    
    return json_data

        
# Helper functions used to parse nested JSON file into a pandas DataFrame.
# Code adapted from:
# https://stackoverflow.com/questions/41180960/convert-nested-json-to-csv-file-in-python
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
    def flatten_json(data, prev_heading=''):
        if isinstance(data, dict):
            rows = [{}]
            for key, value in data.items():
                rows = cross_join(rows, flatten_json(value, key))
        elif isinstance(data, list):
            rows = []
            for i in range(len(data)):
                [rows.append(elem) for elem in flatten_list(flatten_json(data[i], \
                    prev_heading))]
        else:
            rows = [{prev_heading[1:]: data}]
        return rows

    return pd.DataFrame(flatten_json(data_in))


def get_num_sensors_graph(df):
    '''
    Creates a graph showing the number of sensors for each air pollutant in the sensort network
    Saves this graph as an interactive html file in directory and also opens graph in explore page.

    param data_df : type pd.DataFrame : direct dataframe from the get request to the 
                                        London AQI network API
    '''
    # df_num_objectives = df.copy().groupby(by=["SpeciesCode", "ObjectiveName"]).size()
    # print(df_num_objectives)

    # for species in df_num_objectives[df_num_objectives["SpeciesCode"].unique():
    #     print(df_num_objectives[species])
        # num_objectives[species] = df_num_objectives["ObjectiveName"].loc[species].unique()

    # print(df_num_objectives)
    df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']] = df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']].apply(pd.to_datetime)
    df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']] = df[['DateClosed', 'DateOpened', 'DateMeasurementStarted', 'DateMeasurementFinished']].fillna(pd.Timestamp.now())
    df['ActiveDate'] = [pd.date_range(s, e, freq='d') for s, e in zip(pd.to_datetime(df['DateMeasurementStarted']), pd.to_datetime(df['DateMeasurementFinished']))]
    df = df.explode('ActiveDate')
    
    # grouped_df = df.groupby(['SpeciesCode','SpeciesDescription','DateMeasurementStarted', "DateMeasurementFinished"]) \
    #     .size().reset_index(name='Number Of Sensors')
    grouped_df = df.groupby(['SpeciesCode','SpeciesDescription','ActiveDate']).size().reset_index(name='NumberOfSensors')

    # for pollutant in grouped_df["SpeciesCode"]
    #     print(pollutant)
    #     print(grouped_df.loc[pollutant, "Number Of Sensors"])
    #     print(df_num_objectives[pollutant, "Number Of Objectives"])
    #     grouped_df.loc[pollutant, "Number Of Sensors"] /= df_num_objectives[pollutant, "Number Of Objectives"]
    
    
    # grouped_df.to_excel("./data/NumberOfSensors.xlsx")

    fig = px.line(grouped_df, x='ActiveDate', y="NumberOfSensors", color="SpeciesCode", \
        hover_name="SpeciesDescription", title="London Air Quality Sensor Network", markers=True)

    fig.show()

    fig.write_html("./graphs/London_numSensorsGraph.html")


def get_sensor_objectives(df):
    df_2021_data = df[df['Year'] == '2021']

    grouped_df = df_2021_data.groupby(['SpeciesCode','SpeciesDescription', 'ObjectiveName']) \
        .sum().reset_index(name='Number Of Sensors')

    fig = px.bar(grouped_df, x="SpeciesCode", y="Number Of Sensors", color="ObjectiveName", \
        title="Sensor Objectives by Pollutant", orientation='v')
    
    fig.show()

    fig.write_html("./graphs/London_sensorObjectives.html")

def get_data_xl(df):
    '''
    create excel sheet with data, saved to data folder
    param df : pd.DataFrame : dataframe of data
    '''
    df.to_excel("./data/london_sensor_data_1993-2021.xlsx")

London_data = get_air_quality_data("London")
df = json_to_dataframe(London_data)

get_num_sensors_graph(df)

# get_num_objectives_per_species(df)
# get_sensor_objectives(df)
# get_data_xl(df)

