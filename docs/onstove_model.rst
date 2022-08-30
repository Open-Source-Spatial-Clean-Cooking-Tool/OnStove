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
	
   net-benefit = (Morb + Mort + Time + Carb) - (Cap + Fuel + OM)                    \tag{1}

Where; *Morb* is the value of the decrease in morbidity experienced when switching stoves, *Mort* is the value of the decrease in mortality experienced when switching stoves, *Time* is the value of time saved by switching stoves, *Carb* is the value of the decrease in carbon emissions by switching, *Cap* is the capital costs of the stove, *Fuel* is the fuel cost and *OM* is the operation and maintenance cost of the stove. This is a modified version of a prior net-benefit specification by Jeuland et al. [1]_ Each one of the parameters in equation 1 are explained in their respective sub-heading here. 

 .. note::

    The information provided below is first given in general terms. In order to see specific fuel-calculations scroll down to their respective sections

Morbidity and Mortality
-----------------------
The morbidity and mortality terms described the reduced risk of disease and death from four diseases connected to Household Air Pollution (HAP). These diseases are lung cancer (LC), acute lower respiratory infection (ARLI), ischemic heart disease (IHD), chronic obstructive pulmonary disease (COPD) and stroke. The HAP is described in terms of 24-h PM\ :sub:`2.5`\-emissions (measured in µg/m\ :sup:`3`\). Values for of PM\ :sub:`2.5`\ can be found in various different sources. [2]_ [3]_ In OnStove each stove's 24-h PM\ :sub:`2.5`\-emissions is multiplied by an exposure adjustment factor (:math:`\epsilon`). This factor is meant to capture the fact that people tend to change behaviour when acquiring a new stove. The exposure adjusment factor is 0.71 in OnStove as default, this value is typically used for every stove in the analysis except for traditional biomass (in the first application of OnStove a value of 0.51 was used for traditional biomass). This is in line with the work conducted by Das et al. [2]_ Using the adjusted 24-h PM\ :sub:`2.5`\-emissions of each stove the Relative Risk (RR) of contracting LC, ALRI, IHD, COPD and stroke is calculated based on the relation suggested by Burnett et al. [4]_ based on equation 2: 


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




Visualization
*************



References
**********
.. [1] Jeuland, M., Tan Soo, J.-S. & Shindell, D. The need for policies to reduce the costs of cleaner cooking in low income settings: Implications from systematic analysis of costs and benefits. Energy Policy 121, 275–285 (2018).

.. [2] Das, I. et al. The benefits of action to reduce household air pollution (BAR-HAP) model: A new decision support tool. PLOS ONE 16, e0245729 (2021).

.. [3] Dagnachew, A. G., Hof, A. F., Lucas, P. L. & van Vuuren, D. P. Scenario analysis for promoting clean cooking in Sub-Saharan Africa: Costs and benefits. Energy 192, 116641 (2020).

.. [4] Burnett, R. T. et al. An Integrated Risk Function for Estimating the Global Burden of Disease Attributable to Ambient Fine Particulate Matter Exposure. Environmental Health Perspectives 122, 397–403 (2014).

.. [5] IEA, IRENA, UNSD, World Bank & WHO. Tracking SDG 7: The Energy Progress Report. (2022).

.. [6] Stoner, O. et al. Household cooking fuel estimates at global and country level for 1990 to 2030. Nat Commun 12, 5793 (2021).

.. [7] University of Washington. GBD Compare | IHME Viz Hub. http://vizhub.healthdata.org/gbd-compare.