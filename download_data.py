'''
Function to download hourly data for a given site, pollutant, and date range using the following API:
http://api.erg.ic.ac.uk/AirQuality/help/operations/GetRawDataSiteSpeciesCsv

For example, http://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/SiteCode=TD0/SpeciesCode=NO2/StartDate=2007-01-01/EndDate=2008-01-01/csv
will get the hourly data for NO2 at site TD0 from 2007-01-01 to 2008-01-01.
'''
import os
import csv
import pandas as pd
import api_request
from json_to_dataframe import json_to_dataframe
from datetime import datetime, timedelta, date, time
from io import StringIO

def download_species_data(species_code, refresh=False):
    # If we want to refresh our data, delete existing data and logs csv files
    DATA_PATH = f"data/{species_code}_data.csv"
    LOGS_PATH = f"logs/{species_code}_logs.csv"

    if refresh:
        if os.path.isfile(DATA_PATH):
            os.remove(DATA_PATH)
        if os.path.isfile(LOGS_PATH):
            os.remove(LOGS_PATH)

    # Create data and logs csv files
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isfile(DATA_PATH):
        with open(DATA_PATH, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Timestamp", species_code, "SiteCode"])
    
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isfile(LOGS_PATH):
        with open(LOGS_PATH, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["SiteCode", "SpeciesCode", "StartDate", "EndDate", "Timestamp"])

    # Read logs to make sure we don't unnecessarily make a request to the API or
    # download the data twice
    logs_df = pd.read_csv(LOGS_PATH)
    def in_logs(site_code, species_code, start_date, end_date):
        '''
        Helper function which checks the logs to see if we already have data for a given API request.
        '''
        cond_1 = (logs_df['SiteCode'] == site_code)
        cond_2 = (logs_df['SpeciesCode'] == species_code)
        cond_3 = (logs_df['StartDate'] == start_date)
        cond_4 = (logs_df['EndDate'] == end_date)
        return (cond_1 & cond_2 & cond_3 & cond_4).any()

    # Get the date ranges for which sites monitored this pollutant species
    london_json = api_request.get_monitoring_site_species_json()
    london_df = json_to_dataframe(london_json)
    london_species_df = london_df.loc[london_df['SpeciesCode'] == species_code]

    for index, row in london_species_df.iterrows():
        # Get fields for API request
        site_code = row['SiteCode']
        start_date = row['DateMeasurementStarted']
        end_date = row['DateMeasurementFinished']
        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        formatted_start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        formatted_end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")

        request_args = [site_code, species_code, formatted_start_date, formatted_end_date]

        # Check if we've already completed this API request
        if not in_logs(*request_args):
            print(f"Working on data request for {request_args}; ", end="")

            # Get and parse request into pandas dataframe
            data_text = api_request.get_raw_data_site_species_csv(*request_args)
            data_csv_file = StringIO(data_text)
            data_df = pd.read_csv(data_csv_file)

            # Drop empty measurements, add site code as a column, append results to output csv
            data_df.dropna(subset = [data_df.columns[1]], inplace=True)
            data_df['SiteCode'] = site_code
            data_df.to_csv(DATA_PATH, mode="a", index=False, header=False)

            # Add this API request to the logs
            timestamp = datetime.now()
            logs_entry_df = pd.DataFrame([request_args + [timestamp]])
            logs_entry_df.to_csv(LOGS_PATH, mode="a", index=False, header=False)

            print(f"Done @ {timestamp}")

    return

if __name__ == '__main__':
    download_species_data("NO2")
