The OnStove Model
=================
The OnStove determines the net-benefit of different stoves relative to a stated base-line. Net-benefit is in this case defined as all the benefits minus all the costs. The benefits include: reduced morbidity, reduced mortality, reduced emissions and time saved. The costs include: capital costs, fuel costs as well as operational and maintenance costs. The tool has four distinct modules, GIS-processing, baseline calibration, net-benefit calculation and visualization. Each one of these modules are described below in more detail.

GIS data processing
*******************
All the GIS-processing needed for OnStove is conducted in the tool it-self. This part of the algorithm only has to be ran once i.e., if you chose to run several scenarios but keep the GIS-data static, this part does not need to be reran. The tools included in the GIS-processing module of OnStove range from simple geoprocessing tools such as clipping and reprojecting, to more complex tools such as least-cost path algorithms. The tools here are developed to accommodate the necessary OnStove workflows and it should not be viewed as a general GIS-processing tool.


.. note::

    For more information on the different geoprocessing tools included in OnStove see the `Layer scripts <https://onstove-documentation.readthedocs.io/en/latest/layers.html>`_.

Baseline calibration
********************
The baseline calibration is a very important step as all of the stoves included in the analysis are compared relative to the baseline, both with regards to costs and benefits. The baseline calibration does four things: calibrates urban and rural population, calibrates population, calibrates the current stove shares in urban and rural areas and calibrates the electrified population.

**Urban-Rural calibration**

To estimate which areas are urban and which are rural is an important step, as these areas in many industrializing countries tend to have different electrification and clean cooking rates. This step can be done in two ways in OnStove. The user can either create their own urban-rural calibration based on population density or use an external dataset classifying areas into either urban or rural and use that directly.   

.. note::

    See the `GHS calibration <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.onstove.OnStove.calibrate_urban_current_and_future_GHS.html#onstove.onstove.OnStove.calibrate_urban_current_and_future_GHS>`_ and `manual calibration <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.onstove.OnStove.calibrate_urban_manual.html#onstove.onstove.OnStove.calibrate_urban_manual>`_ for more information on the GHS and manual calibration respectively.


**Population calibration**

The population calibration is important as in many instances the geospatial population datasets are outdated. The calibration is carried out across the entire region in order to make sure that the population in the study area matches the value given in the socio-economic file. Furthermore, the calibration ensures that the urban ratio entered in the socio-economic file is also respected. 

.. note::

    For more information on the population calibration see the `Calibrate current pop <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.onstove.OnStove.calibrate_current_pop.html#onstove.onstove.OnStove.calibrate_current_pop>`_.

**Current stove share calibration**

All benefits and costs are relative to the baseline stove calibration. In the technical specs the user enters the urban and rural shares of all stoves that are included in the baseline. The calibration ensures that the urban and rural shares entered in the techno-economic file are respected. All stoves that you wish to include in the calibration need to have the parameters “current_share_rural” and “current_share_urban”. If these two parameters do not exist for a stove it will not be included in the baseline. Note that you do not need to include a stove in the baseline in order to have it as an option in the net-benefit equation. Each stove can also has an “is_base” parameter. This parameter is set to “False” as default, but if it is added in the techno-economic specs as “True” for any stove, everyone in the baseline will cook with this one type of stove (in both urban and rural).

.. note::

    The current stove shares sets the base values to which all benefits and costs are compared, see `set base fuel <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.onstove.OnStove.set_base_fuel.html#onstove.onstove.OnStove.set_base_fuel>`_.


**Current electrification calibration**

The current electrification rate is calibrated using a Multi-Criteria Analysis. The criteria used is population density, nighttime light intensity and proximity to electricity related infrastructure. The electricity infrastructure used is firstly transformers if available, then medium-voltage lines and lastly high-voltage lines. The default is to use equal weights for the three factors included in the calibration, but this can be altered using the socio-economic specification file (using pop_weight, NTL_weight and infra_weight). The electrification calibration is done to ensure that national, urban and rural electrification rates match the values as entered by the user in the socio-economic file. The electrification calibration results in fully, partly, and non electrified settlements. Note that OnStove does not assume an expansion of electricity access i.e., whoever has electricity access currently can cook with electricity and whoever does not, can not. In cases where electrical stoves are the stove with the highest net-benefit in areas without electricity access, the stove with the second highest net-benefit will be chosen.     

Net-benefit calculation
***********************
The net-benefit equation uses includes four benefits (reduced morbidity, reduced mortality, time saved and avoided emissions) as well as three costs (capital cost, fuel cost and Operation and Maintenance (OM) costs). All benefits are monetized in order to be compared to the costs. The net-benefit is defined as all benefits minus all the costs as outlined in equation 1

.. math::

   \mbox{net-benefit } = (Morb + Mort + Time + Carb) - (Cap + Fuel + OM)                    \tag{1}

Where; *Morb* is the value of the decrease in morbidity experienced when switching stoves, *Mort* is the value of the decrease in mortality experienced when switching stoves, *Time* is the value of time saved by switching stoves, *Carb* is the value of the decrease in carbon emissions by switching, *Cap* is the capital costs of the stove, *Fuel* is the fuel cost and *OM* is the operation and maintenance cost of the stove. This is a modified version of a prior net-benefit specification by Jeuland et al. [1]_ Each one of the parameters in equation 1 are explained in their respective sub-heading here. 

 .. note::

    The information provided below is first given in general terms. In order to see specific fuel-calculations scroll down to their respective sections

Morbidity and Mortality
-----------------------
The morbidity and mortality terms described the reduced risk of disease and death from four diseases connected to Household Air Pollution (HAP). These diseases are lung cancer (LC), acute lower respiratory infection (ARLI), ischemic heart disease (IHD), chronic obstructive pulmonary disease (COPD) and stroke. The HAP is described in terms of 24-h PM\ :sub:`2.5`\-emissions (measured in µg/m\ :sup:`3`\). Values of PM\ :sub:`2.5`\ can be found in various different sources. [2]_ [3]_ In OnStove each stove's 24-h PM\ :sub:`2.5`\-emissions is multiplied by an exposure adjustment factor (:math:`\epsilon`). This factor is meant to capture the fact that people tend to change behaviour when acquiring a new stove. The exposure adjusment factor is 0.71 in OnStove as default, this value is typically used for every stove in the analysis except for traditional biomass (in the first application of OnStove a value of 0.51 was used for traditional biomass). This is in line with the work conducted by Das et al. [2]_ Using the adjusted 24-h PM\ :sub:`2.5`\-emissions of each stove the Relative Risk (RR) of contracting LC, ALRI, IHD, COPD and stroke is calculated based on the relation suggested by Burnett et al. [4]_ based on equation 2: 


.. math::
    
    RR = \begin{cases} 
        1, & \mbox{24-h } PM_{2.5}\mbox{-emissions}*\epsilon < z_{rf}
        \\ 1 + \alpha * (1 - \exp(-\beta*(\mbox{24-h } PM_{2.5}\mbox{-emissions}*\epsilon - z_{rf})^\delta)) , & \mbox{24-h } PM_{2.5}\mbox{-emissions}*\epsilon \geq z_{rf}
        \end{cases}

Where; RR is the relative risk associated with each disease studied (LC, IHD, COPD, ALRI and stroke), and α, β, δ and z\ :sub:`rf`\ are disease-specific constants determined experimentally. Note that the equation system indicates that when 24-h PM\ :sub:`2.5`\-emissions are under a certain threshold (z\ :sub:`rf`\) there is no increased risk of disease. The constants α, β, δ and z\ :sub:`rf`\ where determined for each disease by conducting 1,000 runs per disease. For more information on these constants, see Burnett et al. [4]_ and the `data <http://ghdx.healthdata.org/sites/default/files/record-attached-files/IHME_CRCurve_parameters.csv>`_ (clicking the link will download a csv-file). 

Once the RR is determined, the Population Attributable Fraction (PAF) is calculated based on equation 3. PAF is used to express the reduced assess the public health impact when a portion of the population is exposed to a specific risk.

.. math::
    
    \frac{(sfu*(RR_k - 1))}{(sfu*(RR_k - 1) + 1)} = PAF_k \tag{3}


Where; sfu (solid-fuel users) is the share of population not using clean cooking currently and RR\ :sub:`k` is the disease-specific RR determined using equation 2. sfu can be found from e.g. the `IEA website <https://www.iea.org/reports/sdg7-data-and-projections/access-to-clean-cooking>`_, tracking SDG 7 [5]_ or Stoner et al. [6]_ 

Using the PAF calculated with equation 3 the reduced number of cases and deaths per disease can be determined using equations 4 and 5. 

.. math::
    Morb_k = Population * (PAF_0 - PAF_i) * IR_k \tag{4}
.. math::
    Mort_k = Population * (PAF_0 - PAF_i) * MR_k \tag{5}


Where; Population is the total population, MR\ :sub:`k` is the mortality rate associated with the disease and IR\ :sub:`k` is the incidence rate associated with the disease, PAF\ :sub:`0` is the PAF-value for the baseline and PAF\ :sub:`i` is the PAF-value of the new stove. Since PAF\ :sub:`0` and PAF\ :sub:`i` are diversified between urban and rural settlements, so is Morb\ :sub:`k` and Mort\ :sub:`k`. Note that since OnStove is a raster-based geospatial tool the :math:`population` is on a cell-basis. The MR\ :sub:`k` and IR\ :sub:`k` can be diversified by country for each disease (an example source is GBD database [7]_).

The number of cases and deaths avoided are translated to monetary value using the Cost of Illness (COI) and Value of Statistical Life (VSL) (see equations 6 and 7). In cost-benefit analysis, the COI is used to quantify the economic consequences of disease or accidents and the VSL is an important valuation concept in cost-benefit studies, as it is often used as a measure for mortality risk reduction. The equations also include a factor for Cessation Lag for each disease (CL\ :sub:`k`). CL\ :sub:`k` is used to capture the fact that the full health-benefit of switching does not appear instantaneously after a stove-switch. 


.. math::
    Morb = \sum_{k} (\sum_{t=1}^{5} CL_k * COI_k * \frac{Morb_k}{(1+\delta)^{t-1}}) \tag{6}
.. math::
    Mort = \sum_{k} (\sum_{t=1}^{5} CL_k * VSL * \frac{Mort_k}{(1+\delta)^{t-1}}) \tag{7}


Where; CL is the cessation lag (as function of disease k and time t), COI is the cost of illness (as function of disease k), VSL is the value of statistical life, Morb\ :sub:`k` is reduced cases (of disease k), Mort\ :sub:`k` is reduced number of deaths (as result of disease k) and \delta is the discount rate. As the calculations of Morb\ :sub:`k` and Mort\ :sub:`k` (equation 4 and 5) are diversified by cell, so is the values of Morb and Mort.  

.. note::

    In OnStove we assume it takes five years for the full benefits to be experienced. The cessation lags for each disease is hard-coded (see the mobidity and mortality functions). See table 1 for the values currently used in OnStove, these values are in accordance to the values used in BAR-HAP [2]_.

    +---------+-------+-------+------+---------+-------+
    | CL      | COPD  | LC    | IHD  | Stroke  | ALRI  |
    +=========+=======+=======+======+=========+=======+
    | Year 1  | 0.3   | 0.2   | 0.2  | 0.2     | 0.7   |
    +---------+-------+-------+------+---------+-------+
    | Year 2  | 0.2   | 0.1   | 0.1  | 0.1     | 0.1   |
    +---------+-------+-------+------+---------+-------+
    | Year 3  | 0.17  | 0.24  | 0.24 | 0.24    | 0.07  |
    +---------+-------+-------+------+---------+-------+
    | Year 4  | 0.17  | 0.23  | 0.23 | 0.23    | 0.07  |
    +---------+-------+-------+------+---------+-------+
    | Year 5  | 0.16  | 0.23  | 0.23 | 0.23    | 0.06  |
    +---------+-------+-------+------+---------+-------+


Time saved
----------
Each stove has an associated cooking time and an associated collection time. The cooking time and collection times are both entered in the techno-economic specification file (see the `input data section <https://onstove-documentation.readthedocs.io/en/latest/inputs.html#techno-economic-data>`_). The change in time is monetized using the minimum wage in the study area and a geospatial representation of wealth (this can be either a relative wealth index or a poverty layer see the `GIS data section <https://onstove-documentation.readthedocs.io/en/latest/inputs.html#gis-datasets>`_). Similar to the health-benefits, the time-benefits are relative to the baseline. The fuels used for the biomass and biogas stoves are assumed to be collected by the end-users themselves (functions for this are included in OnStove).

**Biomass**

The biomass stoves (both traditional and improved) rely on biomass collected by the end-users themselves. In the first studies using OnStove it has been assumed that the biomass used is firewood. Therefore, a spatial representation of forest cover is used to estimate the time needed to collect fuel (see the `GIS data section <https://onstove-documentation.readthedocs.io/en/latest/inputs.html#gis-datasets>`_). In addition to the forest layer a walking-only friction layer is used. The friction layer describes the time it takes to travel 1 m by foot through each square kilometer [8]_. A spatial least-cost path (in terms of time) is calculated between each settlement and biomass supply sites. The total time spent collecting biomass for cooking would therefore be the traveltime to the site in addition to time needed at the site for the actual collection as outlined in equation 8 (entered in the techno-economic specs file).

**Biogas**

The calculations used for biogas is similar to those for biomass. Biogas is assumed to be produced at a household level by the end-users themselves, who are also the ones collecting the neccesary fuels for its production. In the current version of OnStove manure is assumed to be used to produce biogas. The manure is collected by the households themselves within the square kilometer in which they live. The amount of manure available is estimated with the help of the spatial distribution of livestock (see the `GIS data section <https://onstove-documentation.readthedocs.io/en/latest/inputs.html#gis-datasets>`_), estimates on who much manure each type of animal produces and how much of it can be used for conversion to biogas [9]_. The time needed to collect a sufficient amount of manure is estimated using a walking-only friction layer describing the time it takes to travel 1 m by foot through each square kilometer [8]_. See more information in the documentation for the `biogas class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.Biogas.html>`_. 

Emissions avoided
-----------------

The *Carb* parameter in the net-benefit equation (equation 1), refers to the environmental benefits of reducing greenhouse gas (GHG) emissions. Each stove is assumed to have emissions coupled with its use, and in some cases in the transport or production of its fuel. The value of emissions avoided is calculated using equation 8:

.. math::
    
    Carb = c^{CO_2} * (fueluse_0 * \frac{\gamma_0 * \mu_0}{\epsilon_0} - fueluse_i * \frac{\gamma_i * \mu_i}{\epsilon_i}) \tag{8}

Where; :math:`c^{CO_2}` is the social cost of carbon (USD/tonne) (example source [10]_), :math:`fueluse` is the amount of fuel used for cooking (kWh for electricity, kg for the rest), :math:`\mu` is the energy content of the fuel (MJ/kWh for electricity, MJ/kg for the rest), :math:`\epsilon` is the fuel efficiency of the stove (%), :math:`\gamma` is the carbon intensity of the fuel (kg/GWh for electricity, kg/GJ for the rest) for which five different pollutants (carbon dioxide, methane, carbon monoxide, black carbon and organic carbon) in combination with their 100-year Global Warming Potential (GWP) are used. Subscript :math:`0` denotes the baseline stove combination and, :math:`i` the new stove.

The energy needed to cook a meal is used to estimate :math:`fueluse` for each stove. It is assumed in the current version of OnStove that 3.64 MJ is used to cook a standard meal as outlined by Fuso Nerini et al. [11]_ This value can be changed in onstove.py by changing `self.energy_per_meal`. Using this value, :math:`fueluse` can then be calculated as outlined by equation 9:

.. math::

    \frac{3.64}{\epsilon} *\mu \tag{9}

The carbon intensity :math:`\gamma` of fuel :math:`i`, is calculated according to equation 10.

.. math::
    
    \gamma_i = \sum_{j} \epsilon_{i,j} * GWP_j \tag{10}

Where; Where :math:`\gamma_{(i,j)}` is the emission factor of pollutant :math:`j` of fuel :math:`i` and :math:`GWP_j` the 100-year global warming potential of pollutant :math:`j`.


.. note::

    :math:`\mu`, :math:`\epsilon` and :math:`\gamma` for all stoves except electrical stoves are added in the techno-economic specification file. See fuel specific sections below.

**Biomass**

The carbon emissions caused by the use of woody biomass is dependent by the fraction of Non-Renewable Biomass (fNRB) [12]_. fNRB is defined as the demand of fuelwood that exceeds regrowth in a given area. In the case of biomass equation 10 is modified as outlined in equation 11:

.. math::
    
    \gamma_i = \sum_{j} \epsilon_{i,j} * GWP_j * \psi \mbox{, where } \psi = 1 \mbox{ for } j \neq CO_2  \tag{11}


**Charcoal**

Similar to the case of biomass equation 10 is modified as described in equation 11 when the fuel assessed is charcoal. In addition to this emissions coupled with the production of charcoal are also added to the total emissions. Each kg of charcoal produced is assumed to produce 1,626 g of CO\ :sub:`2`, 255 g of CO, 39.6 g CH\ :sub:`4`, 0.02 g of black carbon and 0.74 g OC [13]_. These values are included in the charcoal class, to change these values refer to the `class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.Charcoal.production_emissions.html>`_.

**LPG**

In addition to stove emissions coupled with LPG-stoves, the transportation of LPG is also assumed to produce emissions. These emissions are dependent on the traveltime needed to transport emissions. The time needed to transport LPG to different settlements is coupled with the assumed emissions of light-commercial vehicles (14 l/h) in order to estimate the total diesel consumption needed for transportation. Each kg of diesel used is assumed to produce 1.52 g of PM (black carbon fraction of PM is assumed to be 0.55 and the OC fraction of black carbon is assumed to be 0.7), 3.169 g of CO\ :sub:`2`, 7.4 g of CO and 0.056 g of N\ :sub:`2`\O. To change these values (as well as the diesel consumption per hour) see the `LPG class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.LPG.transport_emissions.html>`_.

**Electricity**

The production of electricity is coupled with emissions. These emissions are in turn dependent on the grid electricity mix of the study area. The carbon intensity :math:`\gamma_{grid}` is therefore calculated as the weighted average of the emission factors of the generation technologies, see equation 12.

 .. math::
    
    \gamma_{grid} = \frac{\sum_k \epsilon_k * g_k}{\sum_k g_k} \tag{12}

Where; :math:`\gamma_{grid}` is the CO\ :sub:`2`-equivalent intensity of the grid, :math:`\epsilon_k` is the emission factor of generation technology :math:`k` and :math:`g_k` is the electricity generation of technology :math:`k`.

The user is required to enter the installed capacity and power generated by the different powerplants feeding the grid of the study area in order for this calculation to be possible. The emission factors of different powerplants are given in the `Electricity class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.Electricity.html>`_.


Capital cost
------------

The capital cost represents an upfront cost that a user has to pay in order to use a specific stove. The capital cost used in OnStove is investment cost needed for the stove netting out the salvage cost as described in equaton 13.

.. math::

    \mbox{Capital cost } = \mbox{ Investment cost } - \mbox{ Salvage cost} \tag{13}

The salvage cost is assumes a straight-line deprecation as described in equation 14.

.. math::

    \mbox{Salvage cost } = inv * (1 - \frac{\mbox{used life}}{\mbox{technology life}}) * \frac{1}{(1+\delta)^{\mbox{used life}}}  \tag{14}

.. note::

    Values of life times and costs of stoves can be found in various sources e.g. [2]_ [3]_

**LPG**

The cost of buying a refillable LPG-cylinder is added to the investment cost of first time LPG users. Each cylinder is assumed to cost 2.78 USD per kg LPG capacity and the default capacity of the cylinder is assumed to be 12.5 kg of LPG. In addition to this each cylinder is assumed to have a lifetime of 15 years which is taken into account through a salvage cost. These parameters can be changed from the `LPG class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.LPG.html>`_.

**Electricity**

To accomodate for additional capacity needed for electrical cooking it is assumed that the cost of added capacity (as well as its salvage cost) is added to the total capital cost of electricity. The current capacities should be entered in the techno-economic specification file and the life times of technologies in the `Electricity class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.Electricity.html>`_.

Fuel cost
---------

Cost of fuel is important for all stove not assumed to be collected by the end-users themselves. The cost of fuel is divirsified by fuel and the base cost is always entered in the techno-economic specification file.

**Charcoal and pellets**

Charcoal and pellets are assumed to have a fixed cost which is entered in the techno-economic specification file.

**LPG**

The cost of LPG is diversified based on settlement and dependent on the traveltime. In order to estimate the traveltime for LPG to each settlement, OnStove enables two different approaches: 1) to use either LPG vendors or 2) traveltime map directly. For approach 1, a least-cost path between every vendor and settlement is determined. As cost in this case a map visualizing the friction for motorized vehicles is given (see the `GIS data section <https://onstove-documentation.readthedocs.io/en/latest/inputs.html#gis-datasets>`_).  Using the least-cost paths and the vendors a traveltime map for the study area with the vendors as starting points is then calculated. If vendors are not available, approach 2 can be used. Once the traveltime is determined the cost of transporting LPG is determined using an approach similar to described by Szabó et al., [14]_ see equation 15:

.. math::

    \mbox{total costs } = \mbox{LPG costs } + \frac{2 * \mbox{ diesel consumption per h } * \mbox{ LPG costs } * \mbox{ travel time }}{\mbox{Transported LPG}}  \tag{15}

Where; LPG cost is the base cost of LPG. For more information on this calculation refer to `LPG class <https://onstove-documentation.readthedocs.io/en/latest/generated/onstove.technology.LPG.html>`_. 

**Electricity**

The fuel cost associated with electricity is either the grid generation cost or tariff depending which perspective one wish to model from (private or social).

OM cost
-------

Operation and Maintenance cost is assumed to be paid on a yearly basis for all stoves. The costs of this should be entered in the techno-economic specification file as USD per year. Note that having 0 as the OM cost is possible.

Output and Visualization
************************
The outputs of OnStove include a .pkl with all the settlements in the study area and their respective results (e.g., which stove is used where, the investment cost, deaths avoided and health costs avoided). Apart from this .pkl file a summary file is also created (.csv). The .csv file includes rows for each stove in the study area and one line for the total and columns for:

1.  Population (in millions)
2.  Number of households
3.  Total net-benefit (in million USD)
4.  Total deaths avoided (people per year)
5.  Health costs avoided (in million USD)
6.  Time saved (in hours per household and day)
7.  Opportunity cost (in million USD). This is the cost of time speant.
8.  Reduced emissions (in million tonne CO\ :sub:`2`-eq)
9.  Investment cost (in million USD)
10. Fuel cost (in million USD)
11. OM cost (in million USD)
12. Salvage value (in million USD)


There are also several visualization options (see figure below). See the different functions in onstove for more information on what can be plotted using the tool. Note also that all the columns in the .pkl can be extracted and exported using OnStove.

.. figure:: images/main_res_africa.png

    Example OnStove results a)  bar-plot indicating the population stove shares in the scenario, b) spatial distribution of stoves with the highest net-benefit across SSA, c) box-plot indicating the distribution of the net-benefit per household resulting from switching to each stove type and d) total levelized costs and monetized benefits of each stove type.



References
**********
.. [1] Jeuland, M., Tan Soo, J.-S. & Shindell, D. The need for policies to reduce the costs of cleaner cooking in low income settings: Implications from systematic analysis of costs and benefits. Energy Policy 121, 275–285 (2018).

.. [2] Das, I. et al. The benefits of action to reduce household air pollution (BAR-HAP) model: A new decision support tool. PLOS ONE 16, e0245729 (2021).

.. [3] Dagnachew, A. G., Hof, A. F., Lucas, P. L. & van Vuuren, D. P. Scenario analysis for promoting clean cooking in Sub-Saharan Africa: Costs and benefits. Energy 192, 116641 (2020).

.. [4] Burnett, R. T. et al. An Integrated Risk Function for Estimating the Global Burden of Disease Attributable to Ambient Fine Particulate Matter Exposure. Environmental Health Perspectives 122, 397–403 (2014).

.. [5] IEA, IRENA, UNSD, World Bank & WHO. Tracking SDG 7: The Energy Progress Report. (2022).

.. [6] Stoner, O. et al. Household cooking fuel estimates at global and country level for 1990 to 2030. Nat Commun 12, 5793 (2021).

.. [7] University of Washington. GBD Compare | IHME Viz Hub. http://vizhub.healthdata.org/gbd-compare.

.. [8] Weiss, D. J. et al. Global maps of travel time to healthcare facilities. Nat Med 26, 1835–1838 (2020).

.. [9] Lohani, S. P., Dhungana, B., Horn, H. & Khatiwada, D. Small-scale biogas technology and clean cooking fuel: Assessing the potential and links with SDGs in low-income countries – A case study of Nepal. Sustainable Energy Technologies and Assessments 46, 101301 (2021).

.. [10] EPA. Technical Support Document: Social Cost of Carbon, Methane, and Nitrous Oxide: Interim Estimates under Executive Order 13990. 48 (2021).

.. [11] Nerini, F. F., Ray, C. & Boulkaid, Y. The cost of cooking a meal. The case of Nyeri County, Kenya. Environ. Res. Lett. 12, 065007 (2017).

.. [12] Bailis, R., Drigo, R., Ghilardi, A. & Masera, O. The carbon footprint of traditional woodfuels. Nature Clim Change 5, 266–272 (2015).

.. [13] Akagi, S. K. et al. Emission factors for open and domestic biomass burning for use in atmospheric models. https://acp.copernicus.org/preprints/10/27523/2010/acpd-10-27523-2010.pdf (2010) doi:10.5194/acpd-10-27523-2010.

.. [14] Szabó, S., Bódis, K., Huld, T. & Moner-Girona, M. Energy solutions in rural Africa: mapping electrification costs of distributed solar and diesel generation versus grid extension. Environ. Res. Lett. 6, 034002 (2011).