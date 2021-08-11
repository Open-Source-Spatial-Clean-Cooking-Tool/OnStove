import os
import geopandas as gpd
import re
import pandas as pd
import datetime


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


    def set_default_values(self, start_year, end_year, discount_rate):
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

improved_biomass = Technology(tech_life=,
                        inv_cost = 20,
                        om_costs = 1.4,
                        efficiency = 0.33,
                        PM25 = 150)

lpg = Technology(tech_life=,
                        inv_cost = 39,
                        om_costs = 3.56,
                        efficiency = 0.58,
                        PM25 = 10)

biogas = Technology(tech_life=,
                        inv_cost = 430,
                        om_costs = 0.02,
                        efficiency = 0.5)

electricity = Technology(tech_life=,
                        inv_cost = 55,
                        infra_cost =,
                        om_costs = 3.6,
                        efficiency = 0.86)




               

    
