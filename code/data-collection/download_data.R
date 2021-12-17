# Import necessary libraries
library(openair)
library(dplyr)
library(hash)

# Get relative path
path <- getwd()

# Create data directory
data_dir <- paste(path, '/data', sep="")
dir.create(data_dir)

# Get LAQN site and pollutant species codes and information
species_codes_file <- paste(data_dir, '/species.csv', sep="")
monitoring_sites_file <- paste(data_dir, '/monitoring_sites.csv', sep="")
monitoring_site_species_file <- paste(data_dir, '/monitoring_site_species.csv', sep="")

LAQN_pollutants <- read.csv(file=species_codes_file, stringsAsFactors=FALSE)$SpeciesCode
site_codes <- read.csv(file=monitoring_sites_file, stringsAsFactors=FALSE)$SiteCode
monitoring_site_species <- read.csv(monitoring_site_species_file, stringsAsFactors=FALSE)

# Create map of LAQN to openair pollutant codes
LAQN_to_openair_pollutants <- hash()
LAQN_to_openair_pollutants[["CO"]] <- c("co")
LAQN_to_openair_pollutants[["NO2"]] <- c("nox", "no2")
LAQN_to_openair_pollutants[["O3"]] <- c("o3")
LAQN_to_openair_pollutants[["SO2"]] <- c("so2")
LAQN_to_openair_pollutants[["PM10"]] <- c("pm10")
LAQN_to_openair_pollutants[["PM25"]] <- c("pm25")

# Wrap the openair importKCL() function to silently ignore any errors
importKCLSilent <- function(openair_pollutant, site_code, years) {
  data <- tryCatch(
    {na.omit(importKCL(pollutant=openair_pollutant, site=site_code, year=years, meta=TRUE))},
    # TODO: Notify logs of any timeout errors
    error=function(e) NULL)
  return(data)
}

# Use LAQN monitoring site information to get all data from the openair API
for (pollutant in LAQN_pollutants) {
  # TODO: Log successfully downloaded data
  # log_file <- paste(log_dir, '/', pollutant, '_log.csv', sep="")
  # log_cols <- c("SiteCode", "SpeciesCode", "StartDate", "EndDate")
  # if (!file.exists(log_file)) {
  #  file.create(log_file)
  #  write.table(log_cols, log_file)
  # }
  # log <- read.csv(log_file, stringsAsFactors=FALSE)

  # Create data file
  data_file <- paste(data_dir, '/', pollutant, '.csv', sep="")
  
  if (!file.exists(data_file)) {
    # Get relevant sites and date ranges for this pollutant, replacing missing data with NA
    pollutant_info <- filter(monitoring_site_species, SpeciesCode == pollutant)
    pollutant_info[pollutant_info == ""] <- NA
 
    # Function which gets pollutant measurement data based off of LAQN monitoring
    # site information
    getKCLData <- function(row, as.factor=TRUE) {
      site_code <- row["SiteCode"]
      start_year <- format(as.Date(row["DateMeasurementStarted"]),"%Y")
      if (is.na(row["DateMeasurementFinished"])) {
        end_year <- "2021"
      } else {
	end_year <- format(as.Date(row["DateMeasurementFinished"]), "%Y")
      }
      
      # Track progress in console
      current_request <- paste(pollutant, site_code, start_year, end_year, sep=", ")
      print(paste("Getting:", current_request))
      
      # Get and write data
      openair_pollutant <- LAQN_to_openair_pollutants[[pollutant]]
      pollutant_data <- importKCLSilent(openair_pollutant=openair_pollutant, site=site_code, years=start_year:end_year)
      if (!is.null(pollutant_data)) {
        set_cols <- !file.exists(data_file)
        write.table(pollutant_data, file=data_file, sep=",", row.names=FALSE, col.names=set_cols, append=TRUE)
      }
    }
    
    # Apply function
    apply(pollutant_info, 1, getKCLData)
  }
}
