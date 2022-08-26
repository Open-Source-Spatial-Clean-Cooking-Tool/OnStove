The OnStove Model
=================
The OnStove determines the net-benefit of different stoves relative to a stated base-line. Net-benefit is in this case defined as all the benefits minus all the costs. The benefits include: reduced morbidity, reduced mortality, reduced emissions and time saved. The costs include: capital costs, fuel costs as well as operational and maintenance costs. The tool has four distinct modules, GIS-processing, baseline calibration, net-benefit calculation and visualization. Each one of these modules are described below in more detail.

GIS data processing
*******************
All the GIS-processing needed for OnStove is conducted in the tool it-self. This part of the algorithm only has to be ran once i.e., if you chose to run several scenarios but keep the GIS-data static, this part does not need to be reran. The tools included in the GIS-processing module of OnStove range from simple geoprocessing tools such as clipping and reprojecting, to more complex tools such as least-cost path algorithms. The tools here are developed to accommodate the necessary OnStove workflows and it should not be viewed as a general GIS-processing tool.

Baseline calibration
********************
The baseline calibration is a very important step as all of the stoves included in the analysis are compared relative to the baseline, both with regards to costs and benefits. The baseline calibration does four things: calibrates urban and rural population, calibrates population, calibrates the current stove shares in urban and rural areas and calibrates the electrified population.

**Urban-Rural calibration**

To estimate which areas are urban and which are rural is an important step, as these areas in many industrializing countries tend to have different electrification and clean cooking rates. This step can be done in two ways in OnStove. The user can either create their own urban-rural calibration based on population density or use an external dataset classifying areas into either urban or rural and use that directly.   

**Population calibration**

The population calibration is important as in many instances the geospatial population datasets are outdated. The calibration is carried out across the entire region in order to make sure that the population in the study area matches the value given in the socio-economic file. Furthermore, the calibration ensures that the urban ratio entered in the socio-economic file is also respected. 

**Current stove share calibration**

All benefits and costs are relative to the baseline stove calibration. In the technical specs the user enters the urban and rural shares of all stoves that are included in the baseline. The calibration ensures that the urban and rural shares entered in the techno-economic file are respected. All stoves that you wish to include in the calibration need to have the parameters “current_share_rural” and “current_share_urban”. If these two parameters do not exist for a stove it will not be included in the baseline. Note that you do not need to include a stove in the baseline in order to have it as an option in the net-benefit equation. Each stove can also has an “is_base” parameter. This parameter is set to “False” as default, but if it is added in the techno-economic specs as “True” for any stove, everyone in the baseline will cook with this one type of stove (in both urban and rural).


Net-benefit calculation
***********************


Visualization
*************


