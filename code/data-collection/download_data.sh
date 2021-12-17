# Run using: ./download_data.sh
echo "[info] Downloading LAQN data..."
mkdir data
echo "[info] (1/3) Downloading LAQN metadata..."
python3 download_metadata.py
echo "[info] (2/3) Installing required R packages..."
Rscript install_packages.R
echo "[info] (3/3) Downloading LAQN data..."
Rscript download_data.R