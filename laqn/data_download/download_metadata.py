from laqn_api import *
from parse_json import json_to_dataframe

def download_metadata(data_dir:str) -> None:
    """
    Downloads the necessary metadata for the project.

    Args:
        data_dir: Path to data directory.
    """
    json_to_dataframe(get_species_all_json()).to_csv(f"{data_dir}/species.csv", index=False)
    json_to_dataframe(get_monitoring_sites_json()).to_csv(f"{data_dir}/monitoring_sites.csv", index=False)
    json_to_dataframe(get_monitoring_site_species_json()).to_csv(f"{data_dir}/monitoring_site_species.csv", index=False)

if __name__ == "__main__":
    import sys
    download_metadata(data_dir=sys.argv[1])