**********
Quickstart
**********

Software installation
#####################

Requirements
************

Python - Anaconda package
-------------------------

OnStove is written in python, a widely used open-source programming language. Python is a necessary requirement for the OnStove tool to work. Programming in python usually relies on the usage of pre-defined functions that can be found in so called modules. In order to work with OnStove, certain modules need to be installed/updated. The easiest way to do so is by installing Anaconda, a package that contains various useful modules. Anaconda includes all the modules required to run OnStove. Download **Anaconda** `here <https://www.anaconda.com/products/distribution>`_ and install.

.. note::

    * Please make sure that you download the version that is compatible with your operating system (Windows/MacOS/Linux).

    * Following the installation process make sure that you click on the option “Add Python X.X to PATH”. Also by choosing to customize the installation, you can specify the directory of your preference.

    * After the installation you can use the Anaconda command line (search for “Anaconda Prompt”) to run python. It should work by simply writing “python” and pressing enter, since the path has already been included in the system variables. 

Jupyter notebook (via Anaconda)
-------------------------------

Jupyter notebook is a console-based, interactive computing approach providing a web-based application suitable for capturing the whole computation process: developing, documenting, and executing code, as well as communicating the results. Jupyter notebook is used for the online `OnStove example <https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/tree/main/example>`_ included in the official repository , recommended for small analyses and exploring codes and results. Note that Jupyter notebook will be installed automatically with OnStove and therefore no additional action is needed from the user in this step.

Python Interfaces - Integrated Development Environment (IDEs)
-------------------------------------------------------------

Integrated Development Environments are used in order to ease the programming process when multiple or long scripts are required. There are plenty of IDEs developed for Python, the development team behind OnStove has been using PyCharm as their standard IDE. Download **PyCharm** `here <https://www.jetbrains.com/pycharm/>`_ and install.

.. note::

    * Please make sure that you download the version that is compatible with your operating system. Select the “Community”, open-source version.


GIS-software (Optional)
-----------------------

OnStove is a spatial tool and as such, relies on the usage of Geographic Information Systems (GIS). A GIS environment is therefore useful for two main reasons: 1) post-processing purposes and, 2) visualization. Note that all GIS-processes necessary for running an OnStove analysis are included in OnStove itself and pre-processing (as well as visualization) can easily be done without the use of a GIS. The use of QGIS is therefore optional. Download QGIS for free from the official `website <http://www.qgis.org/en/site/>`_.

OnStove installation
********************

To use OnStove the following python modules are needed: `contextily <https://contextily.readthedocs.io/en/latest/>`_, `dill <https://dill.readthedocs.io/en/latest/dill.html>`_, `geopandas <https://geopandas.org/en/stable/>`_, `json <https://docs.python.org/3/library/json.html>`_, `jupyterlab <https://jupyterlab.readthedocs.io/en/stable/>`_, `matplotlib <https://matplotlib.org/>`_, `pandas <https://pandas.pydata.org/>`_, `plotnine <https://plotnine.readthedocs.io/en/stable/>`_, `psycopg2 <https://www.psycopg.org/docs/>`_, `psutil <https://psutil.readthedocs.io/en/latest/>`_, 
`pythondecouple <https://pypi.org/project/python-decouple/>`_, `rasterio <https://rasterio.readthedocs.io/en/latest/>`_, `rasterstats <https://pythonhosted.org/rasterstats/manual.html>`_, `scikitlearn <https://scikit-learn.org/stable/>`_ and `scikitimage <https://scikit-image.org/>`_. These packages are all included in the environment yaml file used for installation. In order to install OnStove follow the steps below. 

1. Go to the official GitHub repository for OnStove `here <https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove>`_.

2. Download the code by going to *Code* and then click *Download Zip*.

3. Once the zip is downloaded, unzip it in a folder of choice.

4. Open your anaconda prompt by searching for *Anaconda prompt* in your computer and double clicking on it. 

5. Navigate to the folder that you downloaded and unzipped by typing *cd PATH/envs*. Where *PATH* is the path on your computer to the unzipped folder. This will bring you to the *envs* folder of the repository.

6. Once in the correct path, type *conda env create --name onstove --file jupyter_env.yaml*. 

.. note::

    There is a file called *jupyter_env.yaml* in the *envs*-folder of the GitHub repository which includes all the python modules needed in order to run OnStove. By typing the command here you create an environment called onstove with all the modules in said .yaml. Installing the modules necessary can take some time. Read more about Anaconda environments `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. 

7. Lastly, still in the anaconda prompt, type *conda activate onstove*. This will take you to your recently created environment and give you access to the packages needed to run OnStove. You are now ready to conduct your own OnStove analysis.


Examples
########
The following section of the manual includes different examples using the OnStove code base in order to show what the code can be used for.

Jupyter example
***************
On the official repository of OnStove there is an example run of OnStove using Jupyter notebook. The notebook downloads the necessary datasets from a Mendeley database and runs an instance of the OnStove. The example goes through the entire workflow from start to finish. This includes GIS-processing, calibration of the baseline, calculating the net-benefits, visualizing and saving the results. The different steps are described more in depth in the notebook. The notebook can be found `here <https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/tree/main/example>`_

Binder example
**************
Try the binder version of OnStove yourself in an online Jupyter notebook:

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Open-Source-Spatial-Clean-Cooking-Tool/OnStove/main?labpath=%2Fexample%2FOnStove_notebook.ipynb
 


