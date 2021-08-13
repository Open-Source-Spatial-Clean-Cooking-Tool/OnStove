import os
import geopandas as gpd
import re
import pandas as pd
import numpy as np
import datetime
from math import exp

from .raster import *


class Technology():
    """
    Template Layer initializing all needed variables.
    """
    def __init__(self,
                 name,
                 tech_life = 0, #in years
                 inv_cost = 0, #in USD
                 infra_cost = 0, # cost of additional infrastructure
                 fuel_cost = 0,
                 time_of_cooking = 0,
                 om_costs = 0, #percentage of investement cost
                 efficiency = 0,#ratio
                 pm25 = 0): # 24-h PM2.5 concentration

        self.name = name
        self.tech_life = tech_life
        self.fuel_cost = fuel_cost
        self.inv_cost = inv_cost
        self.infra_cost = infra_cost
        self.om_costs = om_costs
        self.time_of_cooking = time_of_cooking
        self.efficiency = efficiency
        self.pm25 = pm25


    def set_default_values(start_year, end_year, discount_rate):
        self.discount_rate = discount_rate
        self.start_year = start_year
        self.end_year = end_year

    start_year = 2020
    end_year=2030
    discount_rate = 0.08

Technology.set_default_values(start_year = start_year,
                              end_year = end_year,
                              discount_rate = discount_rate)

traditional_biomass_purchased = Technology(tech_life=3, #placeholder
                        inv_cost = 0.5,
                        om_costs = 138,
                        fuel_cost = 20, #placeholder
                        efficiency = 0.14,
                        pm25 = 500,
                        name = 'purchased_traditional_biomass')

traditional_biomass = Technology(tech_life=3, #placeholder
                        inv_cost = 0.5,
                        om_costs = 138,
                        efficiency = 0.14,
                        pm25 = 500,
                        name = 'traditional_biomass')

improved_biomass = Technology(tech_life=6,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        pm25 = 150,
                        name = 'improved_biomass')

lpg = Technology(tech_life=5,
                        inv_cost = 39,
                        om_costs = 3.56,
                        efficiency = 0.58,
                        pm25 = 10,
                        name = 'lpg')

biogas = Technology(tech_life=5, #placeholder
                        inv_cost = 430,
                        om_costs = 0.02,
                        efficiency = 0.5,
                        name = 'biogas')

electricity = Technology(tech_life=5,
                        inv_cost = 55,
                        infra_cost =500, #placeholder
                        fuel_cost = 0.1, #placeholder
                        om_costs = 3.6,
                        efficiency = 0.86,
                        name = 'electricity')


def morbidity(start_year, end_year, tech, discount_rate, hhsize_R, hhsize_U, sfu = 1):

    """
    Calculates morbidity rate per fuel

    Parameters
    ----------
    arg1 : start_year
        Start year of the analysis
    arg2 : end_year
        End year of the analysis
    arg3: tech
        Stove type assessed
    arg4: discount_rate
        Discount rate to extrapolate costs
    arg5: hhsize_R
        Rural household size
    arg6: hhsize_U
        Urban household size
    arg7: sfu
        Solid fuel users (ration)

    Returns
    ----------
    Monetary morbidity for each stove in urban and rural settings
    """

    if tech.pm25 < 7.298:
        rr_alri = 1
    else:
        rr_alri = 1 + 2.383 * (1 - exp(-0.004 * (tech.pm25 - 7.298) ** 1.193))

    if tech.pm25 < 7.337:
        rr_copd = 1
    else:
        rr_copd = 1 + 22.485 * (1 - exp(-0.001 * (tech.pm25 - 7.337) ** 0.694))

    if tech.pm25 < 7.505:
        rr_ihd = 1
    else:
        rr_ihd = 1 + 2.538 * (1 - exp(-0.081 * (tech.pm25 - 7.505) ** 0.466))

    if tech.pm25 < 7.345:
        rr_lc = 1
    else:
        rr_lc = 1 + 152.496 * (1 - exp(-0.000167 * (tech.pm25 - 7.345) ** 0.76))

    paf_alri = (sfu * (rr_alri - 1)) / (sfu * (rr_alri - 1) + 1)
    paf_copd = (sfu * (rr_copd - 1)) / (sfu * (rr_copd - 1) + 1)
    paf_ihd = (sfu * (rr_ihd - 1)) / (sfu * (rr_ihd - 1) + 1)
    paf_lc = (sfu * (rr_lc - 1)) / (sfu * (rr_lc - 1) + 1)

    coi_alri =
    coi_copd =
    coi_ihd =
    coi_lc =

    morb_alri_U = hhsize_U * paf_alri * incidence_rate_alri
    morb_copd_U = hhsize_U * paf_copd * incidence_rate_copd
    morb_ihd_U = hhsize_U * paf_ihd * incidence_rate_ihd
    morb_lc_U = hhsize_U * paf_lc * incidence_rate_lc

    morb_alri_R = hhsize_R * paf_alri * incidence_rate_alri
    morb_copd_R = hhsize_R * paf_copd * incidence_rate_copd
    morb_ihd_R = hhsize_R * paf_ihd * incidence_rate_ihd
    morb_lc_R = hhsize_R * paf_lc * incidence_rate_lc

    cl_copd = {1:0.3, 2:0.2, 3:0.17, 4:0.17, 5:0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    i = 1
    morb__U_vector = []
    morb__R_vector = []
    while i < 6:
        morbidity_alri_U = cl_alri[i]*coi_alri*morb_alri_U /(1+discount_rate)**(end_year-start_year)
        morbidity_copd_U = cl_copd[i] * coi_copd * morb_copd_U / (1 + discount_rate) ** (end_year - start_year)
        morbidity_lc_U = cl_lc[i] * coi_lc * morb_lc_U / (1 + discount_rate) ** (end_year - start_year)
        morbidity_ihd_U = cl_ihd[i] * coi_ihd * morb_ihd_U / (1 + discount_rate) ** (end_year - start_year)

        morb_U_total = morbidity_alri_U + morbidity_copd_U + morbidity_lc_U + morbidity_ihd_U

        morb__U_vector.append(morb_U_total)

        morbidity_alri_R = cl_alri[i]*coi_alri*morb_alri_R /(1+discount_rate)**(end_year-start_year)
        morbidity_copd_R = cl_copd[i] * coi_copd * morb_copd_R / (1 + discount_rate) ** (end_year - start_year)
        morbidity_lc_R = cl_lc[i] * coi_lc * morb_lc_R / (1 + discount_rate) ** (end_year - start_year)
        morbidity_ihd_R = cl_ihd[i] * coi_ihd * morb_ihd_R / (1 + discount_rate) ** (end_year - start_year)

        morb_R_total = morbidity_alri_R + morbidity_copd_R + morbidity_lc_R + morbidity_ihd_R

        morb__R_vector.append(morb_R_total)

    morbidity_U = np.sum(morb_U_vector)
    morbidity_R = np.sum(morb_R_vector)

    return morbidity_R, morbidity_U


def mortality(start_year, end_year, tech, discount_rate, hhsize_R, hhsize_U, vsl, sfu=1):

    """
    Calculates mortality rate per fuel

    Parameters
    ----------
    arg1 : start_year
        Start year of the analysis
    arg2 : end_year
        End year of the analysis
    arg3: tech
        Stove type assessed
    arg4: discount_rate
        Discount rate to extrapolate costs
    arg5: hhsize_R
        Rural household size
    arg6: hhsize_U
        Urban household size
    arg7: vsl
        Value of statistical life
    arg8: sfu
        Solid fuel users (ration)

    Returns
    ----------
    Monetary mortality for each stove in urban and rural settings
    """

    if tech.pm25 < 7.298:
        rr_alri = 1
    else:
        rr_alri = 1 + 2.383 * (1 - exp(-0.004 * (tech.pm25 - 7.298) ** 1.193))

    if tech.pm25 < 7.337:
        rr_copd = 1
    else:
        rr_copd = 1 + 22.485 * (1 - exp(-0.001 * (tech.pm25 - 7.337) ** 0.694))

    if tech.pm25 < 7.505:
        rr_ihd = 1
    else:
        rr_ihd = 1 + 2.538 * (1 - exp(-0.081 * (tech.pm25 - 7.505) ** 0.466))

    if tech.pm25 < 7.345:
        rr_lc = 1
    else:
        rr_lc = 1 + 152.496 * (1 - exp(-0.000167 * (tech.pm25 - 7.345) ** 0.76))

    paf_alri = (sfu * (rr_alri - 1)) / (sfu * (rr_alri - 1) + 1)
    paf_copd = (sfu * (rr_copd - 1)) / (sfu * (rr_copd - 1) + 1)
    paf_ihd = (sfu * (rr_ihd - 1)) / (sfu * (rr_ihd - 1) + 1)
    paf_lc = (sfu * (rr_lc - 1)) / (sfu * (rr_lc - 1) + 1)

    mort_alri_U = hhsize_U * paf_alri * mortality_rate_alri
    mort_copd_U = hhsize_U * paf_copd * mortality_rate_copd
    mort_ihd_U = hhsize_U * paf_ihd * mortality_rate_ihd
    mort_lc_U = hhsize_U * paf_lc * mortality_rate_lc

    mort_alri_R = hhsize_R * paf_alri * mortality_rate_alri
    mort_copd_R = hhsize_R * paf_copd * mortality_rate_copd
    mort_ihd_R = hhsize_R * paf_ihd * mortality_rate_ihd
    mort_lc_R = hhsize_R * paf_lc * mortality_rate_lc

    cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    i = 1
    mort_U_vector = []
    mort_R_vector = []
    while i < 6:
        mortality_alri_U = cl_alri[i] * vsl * mort_alri_U / (1 + discount_rate) ** (end_year - start_year)
        mortality_copd_U = cl_copd[i] * vsl * mort_copd_U / (1 + discount_rate) ** (end_year - start_year)
        mortality_lc_U = cl_lc[i] * vsl * mort_lc_U / (1 + discount_rate) ** (end_year - start_year)
        mortality_ihd_U = cl_ihd[i] * vsl * mort_ihd_U / (1 + discount_rate) ** (end_year - start_year)

        mort_U_total = mortality_alri_U + mortality_copd_U + mortality_lc_U + mortality_ihd_U

        mort__U_vector.append(mort_U_total)

        mortality_alri_R = cl_alri[i] * vsl * mort_alri_R / (1 + discount_rate) ** (end_year - start_year)
        mortality_copd_R = cl_copd[i] * vsl * mort_copd_R / (1 + discount_rate) ** (end_year - start_year)
        mortality_lc_R = cl_lc[i] * vsl * mort_lc_R / (1 + discount_rate) ** (end_year - start_year)
        mortality_ihd_R = cl_ihd[i] * vsl * mort_ihd_R / (1 + discount_rate) ** (end_year - start_year)

        mort_R_total = mortality_alri_R + mortality_copd_R + mortality_lc_R + mortality_ihd_R

        mort__R_vector.append(mort_R_total)

    mortality_U = np.sum(mort_U_vector)
    mortality_R = np.sum(mort_R_vector)

    return mortality_R, mortality_U


def time_save(tech):

    if tech.name == 'biogas':
        time_of_collection =
    elif tech.name == 'traditional_biomass':
        time_of_collection = 2 * (raster.travel_time(walking_friction, forest)) + 2.2 # 2.2 hrs Medium scenario for Jeiland paper globally, placeholder
    else:
        time_of_collection = 0

        #Read traveltime for biomass (*2) and add time for actual collection












               

    
