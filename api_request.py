'''
Helper functions to download data from the London AQ API:
https://api.erg.ic.ac.uk/AirQuality/Help
'''
import requests
import json
from json_to_dataframe import json_to_dataframe

def get_monitoring_site_species_json(group_name="London"):
    '''
    API GET request to London Open Air Quality API:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetMonitoringSiteSpeciesJson
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName={group_name}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    
    return json_data

def get_raw_data_site_json(site_code, start_date, end_date):
    '''
    API GET request to London Open Air Quality API:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetRawDataSiteJSON
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Data/Site/SiteCode={site_code}/StartDate={start_date}/EndDate={end_date}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

def get_raw_data_site_species_csv(site_code, species_code, start_date, end_date):
    '''
    API GET request to London Open Air Quailty API:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetRawDataSiteSpeciesCsv
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={site_code}/SpeciesCode={species_code}/StartDate={start_date}/EndDate={end_date}/csv"
    response = requests.get(url)
    raw_data = response.text
    return raw_data

def get_monitoring_sites_json(group_name="London"):
    '''
    '''
    url = f"https://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSites/GroupName={group_name}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

def get_species_json():
    '''
    '''
    url = f"https://api.erg.ic.ac.uk/AirQuality/Information/Species/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

if __name__ == '__main__':
    london_json = get_monitoring_sites_json()
    # london_json = get_species_json()
    london_df = json_to_dataframe(london_json)
    # print(london_df.columns)
    
    # london_df.to_csv(f"data/species.csv", mode="w", index=False)

    # london_json = get_monitoring_site_species_json()
    # london_df = json_to_dataframe(london_json)

    site_codes = london_df['SiteCode'].values

    unique_site_codes = set(site_codes)
    print(list(unique_site_codes))
    print(len(unique_site_codes))