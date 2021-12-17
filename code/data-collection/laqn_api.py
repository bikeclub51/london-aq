'''
Helper functions that make GET requests to the LAQN API:
https://api.erg.ic.ac.uk/AirQuality/Help
'''
import json
import requests

def get_species_all_json():
    '''
    GET request to LAQN API for Species Information:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetSpeciesAllJson
    '''
    url = f"https://api.erg.ic.ac.uk/AirQuality/Information/Species/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

def get_monitoring_sites_json(group_name="London"):
    '''
    GET request to LAQN API for Monitoring Sites Information:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetMonitoringSitesJson
    '''
    url = f"https://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSites/GroupName={group_name}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

def get_monitoring_site_species_json(group_name="London"):
    '''
    GET request to LAQN API for Monitoring Site Species information:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetMonitoringSiteSpeciesJson
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName={group_name}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    
    return json_data

def get_raw_data_site_json(site_code, start_date, end_date):
    '''
    GET request to LAQN API for Raw Site Data over a given date range:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetRawDataSiteJSON
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Data/Site/SiteCode={site_code}/StartDate={start_date}/EndDate={end_date}/Json"
    resp = requests.get(url)
    decoded_data = resp.content.decode('utf-8-sig')
    json_data = json.loads(decoded_data)
    return json_data

def get_raw_data_site_species_csv(site_code, species_code, start_date, end_date):
    '''
    GET request to LAQN API for Raw Site + Species Data over a given date range:
    https://api.erg.ic.ac.uk/AirQuality/help/operations/GetRawDataSiteSpeciesCsv
    '''
    url = f"http://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/SiteCode={site_code}/SpeciesCode={species_code}/StartDate={start_date}/EndDate={end_date}/csv"
    response = requests.get(url)
    raw_data = response.text
    return raw_data