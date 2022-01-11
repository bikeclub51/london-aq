# London Air Quality Network Project

## Project Description 
This research project seeks to understand air quality sensor networks within urban environments, using London's Air Quality Network as a case study, and to find the most optimal configuration of air quality sensors through applied Gaussian Process Models and the Optimal Experimental Design Algorithm by Krause et al. 

## Setting Up Workspace
We recommend you use a [conda](https://conda.io/projects/conda/en/latest/index.html) environment to run our code.
Installation instructions can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```
conda create --name london-aq python=3.7
conda activate london-aq
pip3 install jupyter matplotlib pandas numpy requests statsmodels sklearn tensorflow gpflow
pip3 install --upgrade gpflow
conda install r-essentials r-base
```

To exit the environment, run:
```
conda deactivate

## Data Collection
Data including the LAQN sensor locations and temporal/spatial air pollutant conccentrations were accessed through two sources:
    1. London Air Quality Network API: http://api.erg.ic.ac.uk/AirQuality/help
    2. OpenAir Git Library: https://github.com/davidcarslaw/openair 

Data downloaded directly from the LAQN API is stored in [this Dropbox folder] (https://www.dropbox.com/home/Urban%20air%20quality/London%20AQ%20network%20optimization/Fall_2021/Data) given that GitHub cannot store our large data files. For access, please email Professor David Hsu.

Details on how to download the data from the OpenAir Library can be found [here] (https://github.com/bikeclub51/london-aq/tree/main/code/data-collection). 