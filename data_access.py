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

def get_air_quality_data(group_name):
    '''
    API GET request to London Open Air Quality API
    '''
    url = "http://api.erg.ic.ac.uk/AirQuality/Annual/MonitoringObjective/GroupName=" \
        + group_name + "/Json"

    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    
    # print(json_data)
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

    grouped_df = df.groupby(['SpeciesCode','SpeciesDescription','Year']) \
        .size().reset_index(name='Number Of Sensors')

    fig = px.line(grouped_df, x="Year", y="Number Of Sensors", color="SpeciesCode", \
        hover_name="SpeciesDescription", title="London Air Quality Sensor Network", \
            markers=True)

    fig.show()

    fig.write_html("./graphs/London_numSensorsGraph.html")


def get_sensor_objectives(df):
    df_2021_data = df[df['Year'] == '2021']

    grouped_df = df_2021_data.groupby(['SpeciesCode','SpeciesDescription', 'ObjectiveName']) \
        .size().reset_index(name='Number Of Sensors')

    fig = px.bar(grouped_df, x="SpeciesCode", y="Number Of Sensors", color="ObjectiveName", \
        title="Sensor Objectives by Pollutant", orientation='v')
    
    fig.show()

    fig.write_html("./graphs/London_sensorObjectives.html")


London_data = get_air_quality_data("London")
df = json_to_dataframe(London_data)
get_num_sensors_graph(df)
get_sensor_objectives(df)

