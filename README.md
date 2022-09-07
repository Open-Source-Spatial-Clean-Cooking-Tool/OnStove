# <img src="https://user-images.githubusercontent.com/12953752/178504166-47821216-ea94-4241-8b4c-5c6f19a460ec.svg" alt="drawing" style="width:200px"/>

[![Documentation Status](https://readthedocs.org/projects/onstove-documentation/badge/?version=latest)](https://onstove-documentation.readthedocs.io/en/latest/?badge=latest) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/main?labpath=%2Fexample%2FOnStove_notebook.ipynb)

This repository contains the general code for the geospatial cost-benefit clean cooking tool, OnStove. OnStove calculates the net-benefits of different stove options in a given geography and compares all stoves to one another with regards to their net-benefit.

**Introduction**: 

OnStove is developed by the division of Energy Systems at KTH together with partners. The tool is a geospatial, raster-based tool determining the net-benefit of different cooking solutions selected by the user for raster grid cell of a given study area. The tool takes into account four benefits of adopting clean cooking: reduced morbidity, mortality, emissions and time saved, as well as three costs: capital, fuel as well as operation and maintenance (O&M) costs. In each grid cell of the study area the stove with the highest net-benefit is chosen.

OnStove produces scenarios depicting the “true” cost of clean cooking. The scenarios benefits and costs of produced by the tool are to be interpreted as the benefits and costs one could expect if the clean cooking transition was to happen now (overnight change). Results from OnStove are to be interpreted as an upper bound of net-benefits following a switch to cleaner stoves. OnStove can be used by planners and policy makers to identify whether various combinations of interventions in their settings would be worth the potential benefits that could be captured

**Requirements**: 
* [contextily](https://contextily.readthedocs.io/en/latest/)
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
* [rasterstats](https://pythonhosted.org/rasterstats/manual.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [scikit-image](https://scikit-image.org/)

**Installation** 

Through the jupyter_env.yaml in the env-folder using [Anaconda](https://www.anaconda.com/distribution/). 

```
> conda install git
> git clone https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove.git
> cd ..\OnStove\envs
> conda env create --name onstove --file jupyter_env.yaml
> conda activate onstove
```

## Changelog
**22-June-2022**: Original code base (v0.1.0) published

**31-August-2022**: First stable OnStove version (v0.1.1) published

## Resources

### Will be added when journal articles are out

## How to cite

### Will be added when publications come


