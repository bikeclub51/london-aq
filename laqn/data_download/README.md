# Data Download
This module holds the code used to manually download the LAQN data for the project.
We collected the data from the [OpenAir R package](https://github.com/davidcarslaw/openair) created by the Environmental Research Group at King’s College London (KCL). 

We recommend downloading the `data` folder from our [Dropbox](https://www.dropbox.com/sh/udnqd3j1f2y7lb7/AADRbv62hgVegRe1a7GHs5YAa?dl=0) and moving the folder to the root directory. For access, please [email Professor David Hsu](mailto:ydh@mit.edu).

To manually collect data, navigate to this directory and run:
```
$ ./data_download.sh ../../data
```

If this doesn't work, you may need to run:
```
$ chmod +x ./data_download.sh
``` 
first to make the script executable.

# Troubleshooting
1. Error: `sh: 1: /bin/gtar: not found`.
- Fix: Run `$ export TAR="/bin/tar"`.
- For more help, see: https://github.com/r-dbi/RPostgres/issues/110.