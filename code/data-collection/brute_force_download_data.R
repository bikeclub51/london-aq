# Import necessary libraries
library("rstudioapi")
library(openair)
library(dplyr)
library(hash)

# Get relative path
path <- dirname(getSourceEditorContext()$path)

# Create data directory
data_dir <- paste(path, '/brute_force_data', sep="")
dir.create(data_dir)

# Get LAQN site and pollutant species codes
monitoring_sites_file <- paste(data_dir, '/monitoring_sites.csv', sep="")
species_codes_file <- paste(data_dir, '/species.csv', sep="")
monitoring_site_species_file <- paste(data_dir, '/monitoring_site_species.csv', sep="")

site_codes <- read.csv(file=monitoring_sites_file, stringsAsFactors=FALSE)$SiteCode
LAQN_pollutants <- read.csv(file=species_codes_file, stringsAsFactors=FALSE)$SpeciesCode
monitoring_site_species <- read.csv(monitoring_site_species_file, stringsAsFactors=FALSE)

# Create map of LAQN to openair pollutant codes
LAQN_to_openair_pollutants <- hash()
LAQN_to_openair_pollutants[["CO"]] <- c("co")
LAQN_to_openair_pollutants[["NO2"]] <- c("nox", "no2")
LAQN_to_openair_pollutants[["O3"]] <- c("o3")
LAQN_to_openair_pollutants[["SO2"]] <- c("so2")
LAQN_to_openair_pollutants[["PM10"]] <- c("pm10_raw", "pm10", "nv10")
LAQN_to_openair_pollutants[["PM25"]] <- c("v2.5", "nv2.5")

# Wrap the openair importKCL() function to silently ignore any errors
importKCLSilent <- function(openair_pollutant, site_code, years) {
  data <- tryCatch(
    {na.omit(importKCL(pollutant=openair_pollutant, site=site_code, year=years, meta=TRUE))},
    # TODO: Notify logs of any timeout errors
    error=function(e) NULL)
  return(data)
}

# Brute force collection of data
# Iterate through all possible site, pollutant, and year pairs

start_year <- 1987
end_year <- 2021
for (pollutant in api_pollutants) {
  data_file <- paste(data_dir, '/', pollutant, '_final.csv', sep='')
  file.create(data_file)
  set_header <- TRUE

  pollutant_site_codes <- filter(monitoring_site_species, SpeciesCode == pollutant)$SiteCode
  for (site_code in pollutant_site_codes) {
    for (year in start_year:end_year) {

      print(paste(pollutant, site_code, year, sep=", "))
      data <- importKCLSilent(pollutant, site_code, year)
      if (!is.null(data)) {
        write.table(data, file=data_file, sep=",", row.names=FALSE, col.names = set_header, append=TRUE)
        set_header <- FALSE
      }
    }
  }
}
