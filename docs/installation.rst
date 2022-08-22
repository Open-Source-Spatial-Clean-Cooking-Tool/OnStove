Software installation
======================

Requirements
************

**Python - Anaconda package**

OnStove is written in python, a widely used open-source programming language. Python is a necessary requirement for the OnStove tool to work. Programming in python usually relies on the usage of pre-defined functions that can be found in so called modules. In order to work with OnStove, certain modules need to be installed/updated. The easiest way to do so is by installing Anaconda, a package that contains various useful modules. Anaconda includes all the modules required to run OnStove successfully. Download **Anaconda** `here <https://www.continuum.io/downloads>`_ and install.

* Please make sure that you download the version that is compatible with your operating system (Windows/MacOS/Linux - In case you run Windows open the *Windows Control Panel*, go to *System and Security  System* and check e.g. Windows 32-bit or 64-bit).

* Following the installation process make sure that you click on the option “Add Python X.X to PATH”. Also by choosing to customize the installation, you can specify the directory of your preference (suggest something convenient e.g. C:/Python36/..).

* After the installation you can use the Anaconda command line (search for “Anaconda Prompt”) to run python. It should work by simply writing “python” and pressing enter, since the path has already been included in the system variables. In case this doesn’t work, you can either navigate to the specified directory and write “python” there, or add the directory to the PATH by editing the `environment variables <https://www.computerhope.com/issues/ch000549.htm>`_.

**Python Interfaces - Integrated Development Environment (IDEs)**

*	**PyCharm**

Integrated Development Environments are used in order to ease the programming process when multiple or long scripts are required. There are plenty of IDEs developed for Python, KTH-dES has been using PyCharm as the standard IDE to run OnStove.

2. Download **PyCharm** `here <https://www.jetbrains.com/pycharm/>`_ and install.

* Please make sure that you download the version that is compatible with your operating system. Select the “Community”, open-source version.

*	**Jupyter notebook (via Anaconda)**

Jupyter notebook is a console-based, interactive computing approach providing a web-based application suitable for capturing the whole computation process: developing, documenting, and executing code, as well as communicating the results. Jupyter notebook is used for the online OnStove interface, recommended for small analyses and exploring codes and results. Note that Jupyter notebook will be installed automatically with OnStove and therefore no additional action is needed from the user.

**GIS-software (Optional)**

*	**QGIS**

OnSSET is a spatial tool and as such, relies on the usage of Geographic Information Systems (GIS). A GIS environment is therefore useful for two main reasons for post-processing purposes and visualization. Note that all GIS-processes necessary for running an OnStove analysis are included in OnStove itself and pre-processing (as well as visualization) can easliy be done without the use of GIS software. The use of QGIS is therefore optional. Download QGIS for free from the official `website <http://www.qgis.org/en/site/>`_.

OnStove installation
********************

1. Go to the official GitHub repositor for OnStove `here <https://github.com/Open-Source-Spatial-Clean-Cooking-Tool/OnStove>`_.

2. Download the code by going to *Code* and then *Download Zip*.

3. Once the zip is downloaded, unzip it in a folder of choice.

4. Open your anaconda prompt by searching for *Anaconda prompt* in your computer and double clicking on it. 

5. Navigate to the folder that you downloaded and unzipped by typing *cd PATH*. Where *Path* is the path on your computer to the unzipped folder.

6. Once in the correct path, type *conda env create --name onstove --file environment.yml*. There is a folder called *environment.yml* in the GitHub repository which includes all the python modules needed in order to run OnStove. By typing the command here you create an environment called onstove with all the modules in said .yml. Installing the modules necessary can take some time. Read more about Anaconda environments `here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_. 

7. Lastly, still in the anaconda prompt, type *conda activate onstove*. This will take you to your recently created environment and give you access to the packages needed to run OnStove. You are now ready to conduct your own OnStove analysis!


