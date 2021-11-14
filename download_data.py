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
    print(logs_df.shape)

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
    print(london_species_df.shape)

    for index, row in london_species_df.iterrows():
        # Get fields for API request
        site_code = row['SiteCode']
        start_date = row['DateMeasurementStarted']
        end_date = row['DateMeasurementFinished']

        if not end_date:
            end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        # Gets date time type of start and end dates
        dt_start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        dt_end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

        # Formats start and end date back to string
        formatted_start_date = dt_start_date.strftime("%Y-%m-%d")
        formatted_end_date = dt_end_date.strftime("%Y-%m-%d")

        request_args = [site_code, species_code, formatted_start_date, formatted_end_date]
        print(f"Working on data request for {request_args}; ", end="")

        # limit on size of data that can be returned through API, we define it as 3 years
        api_data_limit_years = 3    

        if (dt_end_date.year - dt_start_date.year > api_data_limit_years):
            formatted_temp_start_date = dt_start_date.strftime("%Y-%m-%d")
            
            # iterates over 3 year time ranges between start and end dates
            for year in range(dt_start_date.year + 3, dt_end_date.year, api_data_limit_years):
                dt_temp_end_date = datetime(year, dt_start_date.month, dt_start_date.day)
                formatted_temp_end_date = dt_temp_end_date.strftime("%Y-%m-%d")
                
                request_args = [site_code, species_code, formatted_temp_start_date, formatted_temp_end_date]
                if not in_logs(*request_args):
                    
                    # Make get request on 3 year time limits
                    helper_download_species_data(request_args, DATA_PATH, LOGS_PATH)
                    
                    formatted_temp_start_date = formatted_temp_end_date
            
            # Calls API for missing left over years
            if (dt_temp_end_date < dt_end_date): 
                print("Missing final dates: ", dt_temp_end_date, ", ", dt_end_date)
                formatted_temp_end_date = dt_temp_end_date.strftime("%Y-%m-%d")

                request_args = [site_code, species_code, formatted_temp_end_date, formatted_end_date]

                if not in_logs(*request_args):
                    helper_download_species_data(request_args, DATA_PATH, LOGS_PATH)

        # if time range does not surpass 3 years
        else:

            # Check if we've already completed this API request
            if not in_logs(*request_args):
                helper_download_species_data(request_args, DATA_PATH, LOGS_PATH)

    return

def helper_download_species_data(request_args, DATA_PATH, LOGS_PATH):
    '''
    This function makes the API call to the Open Air API, using valid parameters to API

    RI:
        end_date.year() - start_date.year() <= 5 
    '''
    # Get and parse request into pandas dataframe
    data_text = api_request.get_raw_data_site_species_csv(*request_args)
    data_csv_file = StringIO(data_text)
    data_df = pd.read_csv(data_csv_file)
    # data_df = pd.read_csv(data_csv_file, error_bad_lines=False)

    # Drop empty measurements, add site code as a column, append results to output csv
    data_df.dropna(subset=[data_df.columns[1]], inplace=True)
    
    site_code = request_args[0]
    data_df['SiteCode'] = site_code
    data_df.to_csv(DATA_PATH, mode="a", index=False, header=False)

    # Add this API request to the logs
    timestamp = datetime.now()
    logs_entry_df = pd.DataFrame([request_args+ [timestamp]])
    logs_entry_df.to_csv(LOGS_PATH, mode="a", index=False, header=False)

    print(f"Done @ {timestamp}")


def get_all_site_coordinates():
    '''
    Finds the coordinates location of all site codes using the data from LAQN monitoring site API call.
    param site_code : str : site code of a monitor, from downloaded data in csv
    returns : tuple of floats : coordinates as (latitude, longitude)
    '''
    # function to get coordinates of single inputted site code
    def get_site_coordinates(london_df, site_code):

        latitudes = london_df.loc[london_df['SiteCode'] == site_code]['Latitude']
        latitude = list(set(latitudes))[0]
        longitudes = london_df.loc[london_df['SiteCode'] == site_code]['Longitude']
        longitude = list(set(longitudes))[0]

        return (latitude, longitude)
    
    london_json = api_request.get_monitoring_site_species_json()
    london_df = json_to_dataframe(london_json)

    site_codes = london_df['SiteCode'].values
    unique_site_codes = set(site_codes)
    unique_site_codes_list = list(unique_site_codes)

    site_code_coord_map = {'SiteCode': [], 'Latitude': [], 'Longitude': []}

    for site_code in unique_site_codes_list:
        # get coordinates of site code
        print(site_code)
        coords = get_site_coordinates(london_df, site_code)
        print(coords)
        # add coordinates into the map
        site_code_coord_map["SiteCode"].append(site_code)
        site_code_coord_map["Latitude"].append(coords[0])
        site_code_coord_map["Longitude"].append(coords[1])
    
    print(site_code_coord_map)
    site_code_coord_df = pd.DataFrame(site_code_coord_map)
    site_code_coord_df.to_csv("./data/site_coordinates.csv", mode="a", index=False, header=False)

    
'''
if request_args == ['CT6', 'NO2', '2008-01-01', '2021-10-28']:
    request_args_list = [['CT6', 'NO2', '2008-01-01', '2015-01-01'], 
        ['CT6', 'NO2', '2015-01-01', '2018-01-01'],
        ['CT6', 'NO2', '2018-01-01', '2021-10-28']]
elif request_args == ['CR5', 'NO2', '2008-01-01', '2021-10-28']:
    request_args_list = [['CR5', 'NO2', '2008-01-01', '2015-01-01'], 
        ['CR5', 'NO2', '2015-01-01', '2018-01-01'],
        ['CR5', 'NO2', '2018-01-01', '2021-10-28']]
elif request_args == ['CT6', 'NO2', '2008-01-01', '2015-01-01']:
    request_args_list = [['CT6', 'NO2', '2008-01-01', '2010-01-01'], 
        ['CR5', 'NO2', '2010-01-01', '2013-01-01'],
        ['CR5', 'NO2', '2013-01-01', '2015-01-01']] 
else:
    request_args_list = [request_args]
'''

if __name__ == '__main__':
    # request_args = ['WMC','NO2', '2018-06-20','2021-11-07']
    # data_text = api_request.get_raw_data_site_species_csv(*request_args)
    # print(data_text)
    # download_species_data("O3", refresh=True)
    get_all_site_coordinates()