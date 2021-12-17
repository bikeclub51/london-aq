# London AQ Project
TODO: Add project description

# Setting Up Workspace
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
```