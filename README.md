# London Air Quality Network Project

## Project Description 
This research project seeks to understand air quality sensor networks within urban environments, using London's Air Quality Network as a case study, and to find the most optimal configuration of air quality sensors through applied Gaussian Process Models and the Optimal Experimental Design Algorithm by Krause et al.

## Setting Up Workspace
### 1. Conda Environment

We recommend you use a [conda](https://conda.io/projects/conda/en/latest/index.html) environment to run our code.
Installation instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To initialize an environment with all of the necessary Python packages, run:
```
$ conda create -n london-aq python=3.7
$ conda activate london-aq
$ pip install -r requirements.txt
```

To exit the environment, run:
```
$ conda deactivate
```

Additionally, you will need to have `R` installed and be able to run `Rscript` from the command line if you plan on running the data collection script.

### 2. bayesnewton Update

You will also need to update `bayesnewton/basemodels.py` with the copy of `basemodels.py` that we provide.
To do this, find the location of the file in your conda environment and run:
```
$ cp basemodels.py <PATH_TO_ORIGINAL_FILE>
```

For example, it may look like:
```
$ cp basemodels.py ~/opt/anaconda3/envs/london-aq/lib/python3.7/site-packages/bayesnewton/basemodels.py
```

This will update `basemodels.py` to include a new function, `likelihood_cov()`, which is necessary for our optimization algorithms.

## Data Download
Details on how to download the LAQN sensor data can be found in the `data_download` [module](laqn/data_download/).