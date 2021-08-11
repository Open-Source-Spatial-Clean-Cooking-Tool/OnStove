import os
import geopandas as gpd
import re
import pandas as pd
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
                 om_costs = 0, #percentage of investement cost
                 efficiency = 0,#ratio
                 PM25 = 0): # 24-h PM2.5 concentration

        self.tech_life = tech_life
        fuel_cost = fuel_cost,
        self.inv_cost = inv_cost
        self.infra_cost = infra_cost
        self.om_costs = om_costs
        self.efficiency = efficiency
        self.PM25 = PM25


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

traditional_biomass = Technology(tech_life=3,
                        inv_cost = 0.5,
                        om_costs = 138,
                        efficiency = 0.14,
                        PM25 = 500)

improved_biomass = Technology(tech_life=6,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        PM25 = 150)

lpg = Technology(tech_life=5,
                        inv_cost = 39,
                        om_costs = 3.56,
                        efficiency = 0.58,
                        PM25 = 10)

biogas = Technology(tech_life=5,
                        inv_cost = 430,
                        om_costs = 0.02,
                        efficiency = 0.5)

electricity = Technology(tech_life=5,
                        inv_cost = 55,
                        infra_cost =500, #placeholder
                        om_costs = 3.6,
                        efficiency = 0.86)


def morbidity(tech, discount_rate, sfu, hhsize):
    if tech.PM25 < 7.298:
        rr_alri = 1
    else:
        rr_alri = 1 + 2.383 * (1 - exp(-0.004 * (tech.PM25 - 7.298) ** 1.193))

    if tech.PM25 < 7.337:
        rr_copd = 1
    else:
        rr_copd = 1 + 22.485 * (1 - exp(-0.001 * (tech.PM25 - 7.337) ** 0.694))

    if tech.PM25 < 7.505:
        rr_ihd = 1
    else:
        rr_ihd = 1 + 2.538 * (1 - exp(-0.081 * (tech.PM25 - 7.505) ** 0.466))

    if tech.PM25 < 7.345:
        rr_lc = 1
    else:
        rr_lc = 1 + 152.496 * (1 - exp(-0.000167 * (tech.PM25 - 7.345) ** 0.76))

    paf_alri = (sfu * (rr_alri - 1)) / (sfu * (rr_alri - 1) + 1)
    paf_copd = (sfu * (rr_copd - 1)) / (sfu * (rr_copd - 1) + 1)
    paf_ihd = (sfu * (rr_ihd - 1)) / (sfu * (rr_ihd - 1) + 1)
    paf_lc = (sfu * (rr_lc - 1)) / (sfu * (rr_lc - 1) + 1)

    morb_alri = hhsize * paf_alri * incidence_rate_alri
    morb_copd = hhsize * paf_copd * incidence_rate_copd
    morb_ihd = hhsize * paf_ihd * incidence_rate_ihd
    morb_lc = hhsize * paf_lc * incidence_rate_lc

    


               

    
