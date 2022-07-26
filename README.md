# London Air Quality Network Project

## Project Description 
This research project seeks to understand air quality sensor networks within urban environments, using London's Air Quality Network as a case study, and to find the most optimal configuration of air quality sensors through applied Gaussian Process Models and the Optimal Experimental Design Algorithm by Krause et al.

## Setting Up Workspace
### 1. Conda Environment

We recommend you use a [conda](https://conda.io/projects/conda/en/latest/index.html) environment to run our code.
Installation instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To initialize an environment with all of the necessary Python and R packages, run:
```
$ conda create --name london-aq --file requirements.txt
```
```
python=3.7
conda activate london-aq
pip3 install jupyter matplotlib pandas numpy requests statsmodels sklearn tensorflow gpflow
pip3 install --upgrade gpflow
bayesnewton, objax, jax, jaxlib, numba, dill
```

To exit the environment, run:
```
$ conda deactivate
```
and to reactivate, run:
```
$ conda activate london-aq
```

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