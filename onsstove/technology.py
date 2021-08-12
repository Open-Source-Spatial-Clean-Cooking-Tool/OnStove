import os
import geopandas as gpd
import re
import pandas as pd
import numpy as np
import datetime
from math import exp


class Technology():
    """
    Template Layer initializing all needed variables.
    """
    def __init__(self,
                 tech_life = 0, #in years
                 inv_cost = 0, #in USD
                 infra_cost = 0, # cost of additional infrastructure
                 fuel_cost = 0,
                 time_of_collection = 0,
                 time_of_cooking = 0,
                 om_costs = 0, #percentage of investement cost
                 efficiency = 0,#ratio
                 pm25 = 0): # 24-h PM2.5 concentration

        self.tech_life = tech_life
        self.fuel_cost = fuel_cost,
        self.inv_cost = inv_cost
        self.infra_cost = infra_cost
        self.om_costs = om_costs
        self.efficiency = efficiency
        self.time_of_collection = time_of_collection
        self.time_of_cooking = time_of_cooking
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
                        pm25 = 500)

traditional_biomass = Technology(tech_life=3, #placeholder
                        inv_cost = 0.5,
                        om_costs = 138,
                        efficiency = 0.14,
                        pm25 = 500)

improved_biomass = Technology(tech_life=6,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        pm25 = 150)

lpg = Technology(tech_life=5,
                        inv_cost = 39,
                        om_costs = 3.56,
                        efficiency = 0.58,
                        pm25 = 10)

biogas = Technology(tech_life=5, #placeholder
                        inv_cost = 430,
                        om_costs = 0.02,
                        efficiency = 0.5)

electricity = Technology(tech_life=5,
                        inv_cost = 55,
                        infra_cost =500, #placeholder
                        fuel_cost = 0.1, #placeholder
                        om_costs = 3.6,
                        efficiency = 0.86)


def morbidity(start_year, end_year, tech, discount_rate, sfu, hhsize):

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

    morb_alri = hhsize * paf_alri * 0.5
    morb_copd = hhsize * paf_copd * incidence_rate_copd
    morb_ihd = hhsize * paf_ihd * incidence_rate_ihd
    morb_lc = hhsize * paf_lc * incidence_rate_lc

    cl_copd = {1:0.3, 2:0.2, 3:0.17, 4:0.17, 5:0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    i = 1
    morb_vector = []
    while i < 6:
        morbidity_alri = cl_alri[i]*coi_alri*morb_alri/(1+discount_rate)**(end_year-start_year)
        morbidity_copd = cl_copd[i] * coi_copd * morb_copd / (1 + discount_rate) ** (end_year - start_year)
        morbidity_lc = cl_lc[i] * coi_lc * morb_lc / (1 + discount_rate) ** (end_year - start_year)
        morbidity_ihd = cl_ihd[i] * coi_ihd * morb_ihd / (1 + discount_rate) ** (end_year - start_year)

        morb_total = morbidity_alri + morbidity_copd + morbidity_lc + morbidity_ihd

        morb_vector.append(morb_total)

    morbidity = np.sum(morb_vector)

    return morbidity


def mortality(start_year, end_year, tech, discount_rate, sfu, hhsize, vsl):

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

    mort_alri = hhsize * paf_alri * mortality_rate_alri
    mort_copd = hhsize * paf_copd * mortality_rate_copd
    mort_ihd = hhsize * paf_ihd * mortality_rate_ihd
    mort_lc = hhsize * paf_lc * mortality_rate_lc

    cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    i = 1
    mort_vector = []
    while i < 6:
        mortality_alri = cl_alri[i] * vsl * mort_alri / (1 + discount_rate) ** (end_year - start_year)
        mortality_copd = cl_copd[i] * vsl * mort_copd / (1 + discount_rate) ** (end_year - start_year)
        mortality_lc = cl_lc[i] * vsl * mort_lc / (1 + discount_rate) ** (end_year - start_year)
        mortality_ihd = cl_ihd[i] * vsl * mort_ihd / (1 + discount_rate) ** (end_year - start_year)

        mort_total = mortality_alri + mortality_copd + mortality_lc + mortality_ihd

        mort_vector.append(mort_total)

    mortality = np.sum(mort_vector)

    return mortality

def time_save(tech):

    if tech.time_of_collection == 0:
        saved_time =








               

    
