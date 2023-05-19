# <img src="https://user-images.githubusercontent.com/12953752/178504166-47821216-ea94-4241-8b4c-5c6f19a460ec.svg" alt="drawing" style="width:200px"/>

[![Documentation Status](https://readthedocs.org/projects/onstove-documentation/badge/?version=latest)](https://onstove-documentation.readthedocs.io/en/latest/?badge=latest) 
[![Tests](https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/actions/workflows/tests.yml/badge.svg?event=pull_request)](https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/actions?query=workflow%3Atests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7185177.svg)](https://doi.org/10.5281/zenodo.7185177)

This repository contains the general code for the geospatial cost-benefit clean cooking tool, OnStove. OnStove calculates the net-benefits of different stove options in a given geography and compares all stoves to one another with regards to their net-benefit.

## Introduction 
OnStove is developed by the division of Energy Systems at KTH together with partners. The tool is a geospatial, raster-based tool determining the net-benefit of different cooking solutions selected by the user for raster grid cell of a given study area. The tool takes into account four benefits of adopting clean cooking: reduced morbidity, mortality, emissions and time saved, as well as three costs: capital, fuel as well as operation and maintenance (O&M) costs. In each grid cell of the study area the stove with the highest net-benefit is chosen.

OnStove produces scenarios depicting the “true” cost of clean cooking. The scenarios benefits and costs of produced by the tool are to be interpreted as the benefits and costs one could expect if the clean cooking transition was to happen now (overnight change). Results from OnStove are to be interpreted as an upper bound of net-benefits following a switch to cleaner stoves. OnStove can be used by planners and policy makers to identify whether various combinations of interventions in their settings would be worth the potential benefits that could be captured

## Installation 
Install a python distribution using 
[Anaconda](https://www.anaconda.com/distribution/) or 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html#).

### Installing with `conda`
The easiest way of installing and using `OnStove` is through `conda`. After installing a distribution of `conda`, 
Open an `Anaconda Prompt` or a `Command Prompt` and run:
```
> conda create -n onstove -c conda-forge OnStove
```
Now you will have a new conda environment called `ostove` with `OnStove` installed on it. To use it open a `Command Prompt`
in the root folder of your analysis and activate the enviornment with:
```
> conda activate onstove
```

### Downloading the source code and intalling the environment
Open an `Anaconda Prompt` or a `Command Prompt` and download the source code with:
```
> conda install git
> git clone https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove.git
```
Then use the `jupyter_env.yaml` in the `envs` folder to install the environment by writing:
```
> cd OnStove
> conda env create --name onstove --file envs/jupyter_env.yaml
> conda activate onstove
```

Now your environment `onstove` is available to use. Note that you need to activate it
always before conducting any analysis. 

## Dependencies
`OnStove` relais on the following packages:
* [dill](https://dill.readthedocs.io/en/latest/dill.html)
* [geopandas](https://geopandas.org/en/stable/)
* [jupyterlab](https://jupyterlab.readthedocs.io/en/stable/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [plotnine](https://plotnine.readthedocs.io/en/stable/)
* [psycopg2](https://www.psycopg.org/docs/)
* [psutil](https://psutil.readthedocs.io/en/latest/)
* [python-decouple](https://pypi.org/project/python-decouple/)
* [rasterio](https://rasterio.readthedocs.io/en/latest/)
* [scikit-image](https://scikit-image.org/)
* [svgpathtools](https://pypi.org/project/svgpathtools/)
* [svgpath2mpl](https://pypi.org/project/svgpath2mpl/)

## Documentation
Access the latest documentation in [read the docs](https://readthedocs.org/projects/onstove-documentation/badge/?version=latest).

## Resources

[Publication on sub-Saharan Africa](https://www.nature.com/articles/s41893-022-01039-8)

## How to cite
```
Khavari, Babak, Camilo Ramirez, Marc Jeuland and Francesco Fuso Nerini (12 January 2023). 
"A geospatial approach to understanding clean cooking challenges in sub-Saharan Africa". 
Nature Sustainability. 1–11. ISSN 2398-9629. doi:10.1038/s41893-022-01039-8. 
Creative Commons CC‑BY‑4.0 license.
```

