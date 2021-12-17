# Data Collection
To collect data, navigate to this directory and run:
```
./data_download.sh
```
If this doesn't work you may need to run:
```
chmod +x ./data_download.sh
```
first to make the script executable.
# Known Issues
If you run into the following error:
```
sh: 1: /bin/gtar: not found
```
You may need to run:
```
export TAR="/bin/tar"
```
For more, see: https://github.com/r-dbi/RPostgres/issues/110