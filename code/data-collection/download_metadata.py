'''
Function to download the necessary metadata for the R data download script
'''
from json_to_dataframe import json_to_dataframe
from laqn_api import *

def download_metadata():
    json_to_dataframe(get_species_all_json()).to_csv(f"data/species.csv", index=False)
    json_to_dataframe(get_monitoring_sites_json()).to_csv(f"data/monitoring_sites.csv", index=False)
    json_to_dataframe(get_monitoring_site_species_json()).to_csv(f"data/monitoring_sites_species.csv", index=False)

if __name__ == "__main__":
    download_metadata()