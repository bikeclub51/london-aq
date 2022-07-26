#!/bin/bash
DATA_DIR=$1
echo "[Data] Downloading LAQN data to $DATA_DIR..."
mkdir -p $DATA_DIR
echo "[Data] (1/3) Downloading LAQN metadata..."
python3 download_metadata.py $DATA_DIR
echo "[Data] (2/3) Installing required R packages..."
Rscript install_packages.R
echo "[Data] (3/3) Downloading LAQN data..."
Rscript download_data.R $DATA_DIR