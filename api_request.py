'''
Helper functions to download data from the London AQ API:
https://api.erg.ic.ac.uk/AirQuality/Help
'''
import requests
import json

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