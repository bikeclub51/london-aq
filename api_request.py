'''
Helper functions to download data from the London AQ API:
https://api.erg.ic.ac.uk/AirQuality/Help
'''
import requests
import json

def get_monitoring_site_data(group_name):
  '''
  API GET request to London Open Air Quality API
  '''
  url = f"http://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName={group_name}/Json"
  # url = f"http://api.erg.ic.ac.uk/AirQuality/Daily/MonitoringIndex/Latest/GroupName={group_name}/Json"
  resp = requests.get(url)
  decoded_data = resp.content.decode('utf-8-sig')
  json_data = json.loads(decoded_data)
  
  return json_data

def get_air_quality_data(site_code, start_date, end_date):
  '''
  API GET request to London Open Air Quality API
  '''
  url = f"http://api.erg.ic.ac.uk/AirQuality/Data/Site/SiteCode={site_code}/StartDate={start_date}/EndDate={end_date}/Json"
  resp = requests.get(url)
  decoded_data = resp.content.decode('utf-8-sig')
  json_data = json.loads(decoded_data)
  
  return json_data