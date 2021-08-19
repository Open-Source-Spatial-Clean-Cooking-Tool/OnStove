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
                 name = None,
                 carbon_intensity = 0,
                 energy_content = 0,
                 tech_life = 0, #in years
                 inv_cost = 0, #in USD
                 infra_cost = 0, # cost of additional infrastructure
                 fuel_cost = 0,
                 time_of_cooking = 0,
                 om_costs = 0, #percentage of investement cost
                 efficiency = 0,#ratio
                 pm25 = 0): # 24-h PM2.5 concentration

        self.name = name
        self.carbon_intensity = carbon_intensity
        self.energy_content = energy_content
        self.tech_life = tech_life
        self.fuel_cost = fuel_cost
        self.inv_cost = inv_cost
        self.infra_cost = infra_cost
        self.om_costs = om_costs
        self.time_of_cooking = time_of_cooking
        self.efficiency = efficiency
        self.pm25 = pm25


    def set_default_values(start_year, end_year, discount_rate_social, discount_rate_tech):
        self.discount_rate_social = discount_rate_social
        self.start_year = start_year
        self.end_year = end_year
        self.discount_rate_tech = discount_rate_tech


    start_year = 2020
    end_year=2030
    discount_rate_tech = 0.08
    discount_rate_social = 0.04

Technology.set_default_values(start_year = start_year,
                              end_year = end_year,
                              discount_rate_tech = discount_rate_tech,
                              discount_rate_social = discount_rate_social)

traditional_biomass_purchased = Technology(tech_life=3, #placeholder
                        inv_cost = 0.5,
                        om_costs = 138,
                        fuel_cost = 0.006, #Francecso's paper
                        efficiency = 0.14,
                        pm25 = 500,
                        energy_content = 16, #MJ/kg
                        carbon_intensity = 217, #kg/GJ
                        name = 'purchased_traditional_biomass')

traditional_biomass = Technology(tech_life=3, #placeholder
                        inv_cost = 0.5,
                        om_costs = 138,
                        efficiency = 0.14,
                        pm25 = 500,
                        energy_content = 16, #MJ/kg
                        carbon_intensity = 217, #kg/GJ
                        name = 'traditional_biomass')

improved_biomass = Technology(tech_life=6,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        pm25 = 150,
                        energy_content = 16, #MJ/kg
                        carbon_intensity = 217, #kg/GJ
                        name = 'improved_biomass')

improved_biomass_purchased = Technology(tech_life=6,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        fuel_cost = 0.006, #Francecso's paper
                        pm25 = 150,
                        energy_content = 16, #MJ/kg
                        carbon_intensity = 217, #kg/GJ
                        name = 'improved_biomass_purchased')

lpg = Technology(tech_life=5,
                        inv_cost = 39,
                        om_costs = 3.56,
                        efficiency = 0.58,
                        pm25 = 10,
                        fuel_cost = 0.049,
                        energy_content = 45.5, #MJ/kg
                        carbon_intensity = 67, #kg/GJ
                        name = 'lpg')

biogas = Technology(tech_life=5, #placeholder
                        inv_cost = 430,
                        om_costs = 0.02,
                        efficiency = 0.5,
                        energy_content = 22.8, #MJ/m3
                        carbon_intensity = 4, #kg/GJ
                        name = 'biogas')

electricity = Technology(tech_life=5,
                        inv_cost = 55,
                        infra_cost =500, #placeholder
                        fuel_cost = 0.059, #placeholder
                        om_costs = 3.6,
                        efficiency = 0.86,
                        energy_content = 3.6, # MJ/kWh
                        carbon_intensity = 160, #kg/GWh
                        name = 'electricity')


def morbidity(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, sfu = 1):

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

    paf_alri = (sfu * (rr_alri - 1)) / (sfu * (rr_alri - 1) + 1) #sfu is one.
    paf_copd = (sfu * (rr_copd - 1)) / (sfu * (rr_copd - 1) + 1)
    paf_ihd = (sfu * (rr_ihd - 1)) / (sfu * (rr_ihd - 1) + 1)
    paf_lc = (sfu * (rr_lc - 1)) / (sfu * (rr_lc - 1) + 1)

    coi_alri = 38 #Nepal mean in CCA doc, add as user input
    coi_copd = 37 #Nepal value in CCA doc, add as user input
    coi_ihd = 1010 #mean value in CCA doc, add as user input
    coi_lc = 1010 #mean value in CCA doc, add as user input

    morb_alri_U = hhsize_U * paf_alri * 5 #add hhsize_U and hhsize_R as inputs as user input
    morb_copd_U = hhsize_U * paf_copd * 12
    morb_ihd_U = hhsize_U * paf_ihd * 50
    morb_lc_U = hhsize_U * paf_lc * 7

    morb_alri_R = hhsize_R * paf_alri * 5 #Add incidence rate per disease as input
    morb_copd_R = hhsize_R * paf_copd * 12
    morb_ihd_R = hhsize_R * paf_ihd * 50
    morb_lc_R = hhsize_R * paf_lc * 7

    cl_copd = {1:0.3, 2:0.2, 3:0.17, 4:0.17, 5:0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    i = 1
    morb__U_vector = []
    morb__R_vector = []
    while i < 6:
        morbidity_alri_U = cl_alri[i]*coi_alri*morb_alri_U /(1+discount_rate_social)**(end_year-start_year)
        morbidity_copd_U = cl_copd[i] * coi_copd * morb_copd_U / (1 + discount_rate_social) ** (end_year - start_year)
        morbidity_lc_U = cl_lc[i] * coi_lc * morb_lc_U / (1 + discount_rate_social) ** (end_year - start_year)
        morbidity_ihd_U = cl_ihd[i] * coi_ihd * morb_ihd_U / (1 + discount_rate_social) ** (end_year - start_year)

        morb_U_total = morbidity_alri_U + morbidity_copd_U + morbidity_lc_U + morbidity_ihd_U

        morb__U_vector.append(morb_U_total)

        morbidity_alri_R = cl_alri[i]*coi_alri*morb_alri_R /(1+discount_rate_social)**(end_year-start_year)
        morbidity_copd_R = cl_copd[i] * coi_copd * morb_copd_R / (1 + discount_rate_social) ** (end_year - start_year)
        morbidity_lc_R = cl_lc[i] * coi_lc * morb_lc_R / (1 + discount_rate_social) ** (end_year - start_year)
        morbidity_ihd_R = cl_ihd[i] * coi_ihd * morb_ihd_R / (1 + discount_rate_social) ** (end_year - start_year)

        morb_R_total = morbidity_alri_R + morbidity_copd_R + morbidity_lc_R + morbidity_ihd_R

        morb__R_vector.append(morb_R_total)

    morbidity_U = np.sum(morb_U_vector)
    morbidity_R = np.sum(morb_R_vector)

    return morbidity_R, morbidity_U


def mortality(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, vsl, sfu=1):

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

    mort_alri_U = hhsize_U * paf_alri * 22 #Mortality rate per country --> add as input
    mort_copd_U = hhsize_U * paf_copd * 95
    mort_ihd_U = hhsize_U * paf_ihd * 69
    mort_lc_U = hhsize_U * paf_lc * 13

    mort_alri_R = hhsize_R * paf_alri * 22
    mort_copd_R = hhsize_R * paf_copd * 95
    mort_ihd_R = hhsize_R * paf_ihd * 69
    mort_lc_R = hhsize_R * paf_lc * 13

    cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
    cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
    cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
    cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

    #vsl = 45000 # South Asia mean accroding to CCA doc, add as input

    i = 1
    mort_U_vector = []
    mort_R_vector = []
    while i < 6:
        mortality_alri_U = cl_alri[i] * vsl * mort_alri_U / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_copd_U = cl_copd[i] * vsl * mort_copd_U / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_lc_U = cl_lc[i] * vsl * mort_lc_U / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_ihd_U = cl_ihd[i] * vsl * mort_ihd_U / (1 + discount_rate_social) ** (end_year - start_year)

        mort_U_total = mortality_alri_U + mortality_copd_U + mortality_lc_U + mortality_ihd_U

        mort__U_vector.append(mort_U_total)

        mortality_alri_R = cl_alri[i] * vsl * mort_alri_R / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_copd_R = cl_copd[i] * vsl * mort_copd_R / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_lc_R = cl_lc[i] * vsl * mort_lc_R / (1 + discount_rate_social) ** (end_year - start_year)
        mortality_ihd_R = cl_ihd[i] * vsl * mort_ihd_R / (1 + discount_rate_social) ** (end_year - start_year)

        mort_R_total = mortality_alri_R + mortality_copd_R + mortality_lc_R + mortality_ihd_R

        mort__R_vector.append(mort_R_total)

    mortality_U = np.sum(mort_U_vector)
    mortality_R = np.sum(mort_R_vector)

    return mortality_R, mortality_U

def time_save(tech, value_of_time, walking_friction, forest):

    if tech.name == 'biogas':
        time_of_collection = 2
    elif tech.name == 'traditional_biomass' or tech.name == 'improved_biomass':
        time_of_collection = 2 * (raster.travel_time(walking_friction, forest)) + 2.2 # 2.2 hrs Medium scenario for Jeiland paper globally, placeholder
    else:
        time_of_collection = 0

    time = time_of_collection + tech.time_of_cooking
    time_value = time * value_of_time

    return time_value


def carbon_emissions(tech):

    carb = 5 * (3.64 / tech.efficiency) / tech.energy_content * (tech.carbon_intensity * tech.energy_content / tech.efficiency) #5 USD/MT is average social cost of carbon emissions in Nepal according to https://www.nature.com/articles/s41558-018-0282-y.pdf, 3.64 MJ to cook based on https://iopscience.iop.org/article/10.1088/1748-9326/aa6fd0/meta

    return carb

def discount_factor(discount_rate_tech, tech):
    if tech.start_year == tech.end_year:
        proj_life = 1
    else:
        proj_life = tech.end_year - tech.start_year

    year = np.arange(proj_life)

    discount_factor = (1 + discount_rate_tech) ** year

    return discount_factor, proj_life


def discounted_meals(meals_per_year, discount_rate_tech, tech):
    discount_factor, proj_life = discount_factor(discount_rate_tech, tech)

    energy = meals_per_year * 3.64 / tech.efficiency

    energy_needed = energy * np.ones(proj_life)

    if proj_life > 1:
        energy_needed[0] = 0

    discounted_energy = energy_needed / discount_factor

    return discounted_energy


def discounted_inv(discount_rate_tech, tech):
    discount_factor, proj_life = discount_factor(discount_rate_tech, tech)

    investments = np.zeros(project_life)
    investments[0] = tech.inv_cost

    if proj_life > tech.tech_life:
        investments[tech.tech_life] = tech.inv_cost

    discounted_investments = investments / discount_factor

    return discounted_investments


def discounted_om(discount_rate_tech, tech):
    discount_factor, proj_life = discount_factor(discount_rate_tech, tech)

    operation_and_maintenance = tech.om_costs * np.ones(project_life)

    if proj_life > 1:
        operation_and_maintenance[0] = 0

        if proj_life > tech.tech_life:
            operation_and_maintenance[tech.tech_life] = 0

    discounted_om_cost = operation_and_maintenance / discount_factor

    return discounted_om_cost


def salvage(discount_rate_tech, tech):
    discount_factor, proj_life = discount_factor(discount_rate_tech, tech)

    salvage = np.zeros(project_life)
    used_life = project_life

    if proj_life > tech.tech_life:
        used_life = project_life - tech.tech_life

    salvage[-1] = tech.inv_cost * (1 - used_life / tech.tech_life)

    discounted_salvage = salvage / discount_factor

    return discounted_salvage


def discounted_fuel_cost(discount_rate_tech, tech, road_friction, lpg, meals_per_year):

    discount_factor, proj_life = discount_factor(discount_rate_tech, tech)

    if tech.name == 'electricity' or tech.name == 'improved_biomass_purchased' or tech.name == 'purchased_traditional_biomass':
        fuel_cost = (tech.fuel_cost * discounted_meals(meals_per_year, discount_rate_tech, tech)) * np.ones(project_life) / tech.efficiency
    elif tech.name == 'lpg':
        fuel_cost = (tech.fuel_cost * discounted_meals(meals_per_year, discount_rate_tech, tech)) * raster.lpg_transportation_cost(raster.travel_time(road_friction, lpg)) / tech.efficiency
    else:
        fuel_cost = 0

    fuel_cost_discounted = fuel_cost / discount_factor

    return fuel_cost_discounted

def cost(discount_rate_tech, tech, meals_per_year, road_friction, lpg):

    cost = (discounted_inv(discount_rate_tech, tech) + discounted_om(discount_rate_tech, tech) + \
            discounted_fuel_cost(discount_rate_tech, tech, road_friction, lpg) - salvage(discount_rate_tech, tech))/ \
           discounted_meals(meals_per_year, discount_rate_tech, tech)

    return cost

def benefit(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, vsl,  value_of_time, walking_friction, forest, sfu = 1):

    benefit = morbidity(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, sfu) + \
              mortality(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, vsl, sfu) + \
              time_save(tech, value_of_time, walking_friction, forest) + carbon_emissions(tech)

    return benefit

def lccm(discount_rate_tech, tech, meals_per_year, road_friction, lpg,start_year, end_year, discount_rate_social,
         hhsize_R, hhsize_U, vsl,  value_of_time, walking_friction, forest, sfu = 1):

    lccm = benefit(start_year, end_year, tech, discount_rate_social, hhsize_R, hhsize_U, vsl,  value_of_time,
                   walking_friction, forest, sfu) - cost(discount_rate_tech, tech, meals_per_year, road_friction, lpg)

    return lccm











