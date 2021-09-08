import os
import geopandas as gpd
import re
import pandas as pd
import numpy as np
import datetime
from math import exp

from .raster import *
from .layer import *


class Technology:
    """
    Standard technology class.
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=0,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 infra_cost=0,  # cost of additional infrastructure
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=0,
                 is_base=False,
                 transport_cost=0):  # 24-h PM2.5 concentration

        self.name = name
        self.carbon_intensity = carbon_intensity
        self.energy_content = energy_content
        self.tech_life = tech_life
        self.fuel_cost = fuel_cost
        self.inv_cost = inv_cost
        self.infra_cost = infra_cost
        self.om_cost = om_cost
        self.time_of_cooking = time_of_cooking
        self.efficiency = efficiency
        self.pm25 = pm25
        self.time_of_collection = None
        self.fuel_use = None
        self.is_base = is_base
        self.transport_cost = transport_cost

    def __setitem__(self, idx, value):
        if idx == 'name':
            self.name = value
        elif idx == 'energy_content':
            self.energy_content = value
        elif idx == 'carbon_intensity':
            self.carbon_intensity = value
        elif idx == 'fuel_cost':
            self.fuel_cost = value
        elif idx == 'tech_life':
            self.tech_life = value
        elif idx == 'inv_cost':
            self.inv_cost = value
        elif idx == 'infra_cost':
            self.infra_cost = value
        elif idx == 'om_cost':
            self.om_cost = value
        elif idx == 'time_of_cooking':
            self.time_of_cooking = value
        elif idx == 'efficiency':
            self.efficiency = value
        elif idx == 'pm25':
            self.pm25 = value
        elif idx == 'time_of_collection':
            self.time_of_collection = value
        elif idx == 'fuel_use':
            self.fuel_use = value
        elif idx == 'is_base':
            self.is_base = value
        else:
            raise KeyError(idx)

    def relative_risk(self):
        if self.pm25 < 7.298:
            rr_alri = 1
        else:
            rr_alri = 1 + 2.383 * (1 - exp(-0.004 * (self.pm25 - 7.298) ** 1.193))

        if self.pm25 < 7.337:
            rr_copd = 1
        else:
            rr_copd = 1 + 22.485 * (1 - exp(-0.001 * (self.pm25 - 7.337) ** 0.694))

        if self.pm25 < 7.505:
            rr_ihd = 1
        else:
            rr_ihd = 1 + 2.538 * (1 - exp(-0.081 * (self.pm25 - 7.505) ** 0.466))

        if self.pm25 < 7.345:
            rr_lc = 1
        else:
            rr_lc = 1 + 152.496 * (1 - exp(-0.000167 * (self.pm25 - 7.345) ** 0.76))

        return rr_alri, rr_copd, rr_ihd, rr_lc

    def paf(self, rr, sfu):

        paf = (sfu * (rr - 1)) / (sfu * (rr - 1) + 1)

        return paf

    @staticmethod
    def discount_factor(specs_file):
        '''

        :param self:
        :param specs_file: social specs file
        :return: discount factor to be used for all costs in the net benefit fucntion and the years of analysis
        '''
        if specs_file["Start_year"] == specs_file["End_year"]:
            proj_life = 1
        else:
            proj_life = specs_file["End_year"] - specs_file["Start_year"]

        year = np.arange(proj_life)

        discount_factor = (1 + specs_file["Discount_rate_tech"]) ** year

        return discount_factor, proj_life

    def carb(self):
        self.carbon = (3.64 / self.efficiency) / self.energy_content * (
                    self.carbon_intensity * self.energy_content / self.efficiency)

    def carbon_emissions(self, specs_file, carb_base_fuel):
        self.carb()
        carbon = specs_file["Cost of carbon emissions"] * (carb_base_fuel - self.carbon)

        self.decreased_carbon_emissions = carbon

    def mortality(self, specs_file, gdf, paf_0_alri, paf_0_copd, paf_0_lc, paf_0_ihd):
        """
        Calculates mortality rate per fuel

        Returns
        ----------
        Monetary mortality for each stove in urban and rural settings
        """

        rr_alri, rr_copd, rr_ihd, rr_lc = self.relative_risk()

        paf_alri = self.paf(rr_alri, 1 - specs_file['clean_cooking_access'])
        paf_copd = self.paf(rr_copd, 1 - specs_file['clean_cooking_access'])
        paf_ihd = self.paf(rr_ihd, 1 - specs_file['clean_cooking_access'])
        paf_lc = self.paf(rr_lc, 1 - specs_file['clean_cooking_access'])

        mort_alri = specs_file["Population_start_year"] * (paf_0_alri - paf_alri) * (specs_file["Mort_ALRI"] / 100000)
        mort_copd = specs_file["Population_start_year"] * (paf_0_copd - paf_copd) * (specs_file["Mort_COPD"] / 100000)
        mort_ihd = specs_file["Population_start_year"] * (paf_0_ihd - paf_ihd) * (specs_file["Mort_IHD"] / 100000)
        mort_lc = specs_file["Population_start_year"] * (paf_0_lc - paf_lc) * (specs_file["Mort_LC"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        mort_vector = []
        while i < 6:
            mortality_alri = cl_alri[i] * specs_file["VSL"] * mort_alri / (1 + specs_file["Discount_rate"]) ** (i - 1)
            mortality_copd = cl_copd[i] * specs_file["VSL"] * mort_copd / (
                    1 + specs_file["Discount_rate"]) ** (i - 1)
            mortality_lc = cl_lc[i] * specs_file["VSL"] * mort_lc / (
                    1 + specs_file["Discount_rate"]) ** (i - 1)
            mortality_ihd = cl_ihd[i] * specs_file["VSL"] * mort_ihd / (
                    1 + specs_file["Discount_rate"]) ** (i - 1)

            mort_total = (1 + specs_file["Health_spillovers_parameter"]) * (
                    mortality_alri + mortality_copd + mortality_lc + mortality_ihd)

            mort_vector.append(mort_total)

            i += 1

        mortality = np.sum(mort_vector)

        self.distributed_mortality = gdf["Calibrated_pop"] / gdf["Calibrated_pop"].sum() * mortality
        self.mortality = mortality
        self.deahts_avoided = (mort_alri + mort_copd + mort_lc + mort_ihd)

    def morbidity(self, specs_file, gdf, paf_0_alri, paf_0_copd, paf_0_lc, paf_0_ihd):
        """
        Calculates morbidity rate per fuel

        Returns
        ----------
        Monetary morbidity for each stove in urban and rural settings
        """

        rr_alri, rr_copd, rr_ihd, rr_lc = self.relative_risk()

        paf_alri = self.paf(rr_alri, 1 - specs_file['clean_cooking_access'])
        paf_copd = self.paf(rr_copd, 1 - specs_file['clean_cooking_access'])
        paf_ihd = self.paf(rr_ihd, 1 - specs_file['clean_cooking_access'])
        paf_lc = self.paf(rr_lc, 1 - specs_file['clean_cooking_access'])

        morb_alri = specs_file["Population_start_year"] * (paf_0_alri - paf_alri) * (specs_file["Morb_ALRI"] / 100000)
        morb_copd = specs_file["Population_start_year"] * (paf_0_copd - paf_copd) * (specs_file["Morb_COPD"] / 100000)
        morb_ihd = specs_file["Population_start_year"] * (paf_0_ihd - paf_ihd) * (specs_file["Morb_IHD"] / 100000)
        morb_lc = specs_file["Population_start_year"] * (paf_0_lc - paf_lc) * (specs_file["Morb_LC"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        morb_vector = []
        while i < 6:
            morbidity_alri = cl_alri[i] * specs_file["COI_ALRI"] * morb_alri / (1 + specs_file["Discount_rate"]) ** (
                        i - 1)
            morbidity_copd = cl_copd[i] * specs_file["COI_COPD"] * morb_copd / (1 + specs_file["Discount_rate"]) ** (
                        i - 1)
            morbidity_lc = cl_lc[i] * specs_file["COI_LC"] * morb_lc / (1 + specs_file["Discount_rate"]) ** (i - 1)
            morbidity_ihd = cl_ihd[i] * specs_file["COI_IHD"] * morb_ihd / (1 + specs_file["Discount_rate"]) ** (i - 1)

            morb_total = (1 + specs_file["Health_spillovers_parameter"]) * (
                    morbidity_alri + morbidity_copd + morbidity_lc + morbidity_ihd)

            morb_vector.append(morb_total)

            i += 1

        morbidity = np.sum(morb_vector)

        self.distributed_morbidity = gdf["Calibrated_pop"] / gdf["Calibrated_pop"].sum() * morbidity
        self.morbidity = morbidity
        self.cases_avoided = (morb_alri + morb_copd + morb_lc + morb_ihd)

    def salvage(self, specs_file):
        """
        Calculates discounted salvage cost assuming straight-line depreciation
        Returns
        ----------
        discounted salvage cost
        """
        discount_rate, proj_life = self.discount_factor(self, specs_file)
        salvage = np.zeros(proj_life)
        used_life = proj_life % self.tech_life

        salvage[-1] = self.inv_cost * (1 - used_life / self.tech_life)

        discounted_salvage = salvage.sum() / discount_rate

        self.discounted_salvage_cost = discounted_salvage

    def discounted_om(self, specs_file):
        """
        Calls discount_factor function and creates discounted OM costs.
        Returns
        ----------
        discountedOM costs for each stove during the project lifetime
        """
        discount_rate, proj_life = self.discount_factor(self, specs_file)

        operation_and_maintenance = self.om_costs * np.ones(proj_life) * self.inv_cost
        operation_and_maintenance[0] = 0

        i = self.tech_life
        while i < proj_life:
            operation_and_maintenance[i] = 0
            i = i + self.tech_life

        discounted_om_cost = operation_and_maintenance.sum() / discount_rate

        self.discounted_om_costs = discounted_om_cost

    def discounted_inv(self, specs_file):
        """
        Calls discount_factor function and creates discounted investment cost. Uses proj_life and tech_life to determine
        number of necessary re-investments

        Returns
        ----------
        discounted investment cost for each stove during the project lifetime
        """
        discount_rate, proj_life = self.discount_factor(self, specs_file)

        investments = np.zeros(proj_life)
        investments[0] = self.inv_cost

        i = self.tech_life
        while i < proj_life:
            investments[i] = self.inv_cost
            i = i + self.tech_life

        discounted_investments = investments.sum() / discount_rate

        self.discounted_investments = discounted_investments

    def discounted_meals(self, specs_file):
        discount_rate, proj_life = self.discount_factor(specs_file)

        energy = specs_file["Meals_per_day"] * 365 * 3.64 / self.efficiency

        energy_needed = energy * np.ones(proj_life)

        self.discounted_energy = energy_needed / discount_rate

    def discounted_fuel_cost(self, specs_file):

        discount_rate, proj_life = self.discount_factor(specs_file)

        energy = specs_file["Meals_per_day"] * 365 * 3.64 / self.efficiency

        fuel_cost = np.ones(proj_life) * (energy * (self.fuel_cost / self.energy_content) + self.transport_cost)

        fuel_cost_discounted = fuel_cost.sum() / discount_rate

        self.discounted_fuel_cost = fuel_cost_discounted

    def total_time(self, specs_file):
        self.total_time_yr = self.time_of_cooking * specs_file['Meals_per_day'] * 365

    def time_saved(self, df, specs_file):
        self.total_time(specs_file)
        self.total_time_saved = df["base_fuel_time"] - self.total_time_yr  # time saved per household

        self.time_value = self.total_time_saved * df["value_of_time"] * df[
            "Households"]  # time value of time saved per sq km

    def costs(self):
        self.costs = (
                                 self.discounted_fuel_cost + self.discounted_inv + self.discounted_om_costs - self.discounted_salvage_cost) / self.discounted_energy

    def net_benefit(self, df):

        df["net_benefit_{}".fromat(self.name)] = df.apply(lambda
                                                              row: self.urban_morbidity + self.urban_mortality + self.decreased_carbon_emissions + self.time_value - self.costs if
        df["IsUrban"] == 2 else
        self.rural_morbidity + self.rural_mortality + self.decreased_carbon_emissions + self.time_value - self.costs)


class LPG(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=0,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 infra_cost=0,  # cost of additional infrastructure
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=0,
                 travel_time=None,
                 truck_capacity=2000,
                 diesel_price=0.88,
                 diesel_per_hour=14):
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, infra_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
        self.travel_time = travel_time
        self.truck_capacity = truck_capacity
        self.diesel_price = diesel_price
        self.diesel_per_hour = diesel_per_hour
        self.transport_cost = None

    def transportation_cost(self, specs_file):
        """The cost of transporting LPG. See https://iopscience.iop.org/article/10.1088/1748-9326/6/3/034002/pdf for the formula

        Transportation cost = (2 * diesel consumption per h * national diesel price * travel time)/transported LPG

        Total cost = (LPG cost + Transportation cost)/efficiency of LPG stoves


        Each truck is assumed to transport 2,000 kg LPG
        (3.5 MT truck https://www.wlpga.org/wp-content/uploads/2019/09/2019-Guide-to-Good-Industry-Practices-for-LPG-Cylinders-in-the-
        Distribution-Channel.pdf)
        National diesel price in Nepal is assumed to be 0.88 USD/l
        Diesel consumption per h is assumed to be 14 l/h (14 l/100km)
        (https://www.iea.org/reports/fuel-consumption-of-cars-and-vans)
        LPG cost in Nepal is assumed to be 19 USD per cylinder (1.34 USD/kg)
        LPG stove efficiency is assumed to be 60%

        :param param1:  travel_time_raster
                        Hour to travel between each point and the startpoints as array
        :returns:       The cost of LPG in each cell per kg
        """
        transport_cost = (2 * self.diesel_per_hour * self.diesel_price * self.travel_time) / self.truck_capacity
        kg_yr = (specs_file["Meals_per_day"] * 365 * 3.64) / (self.efficiency * self.energy_content)  # energy content in MJ/kg
        self.transport_cost = transport_cost * kg_yr

    def discounted_fuel_cost(self, specs_file):
        self.transportation_cost(specs_file)
        super().discounted_fuel_cost(specs_file)


class Biomass(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=0,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 infra_cost=0,  # cost of additional infrastructure
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=0,
                 travel_time=None):
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, infra_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
        self.travel_time = travel_time

    def transportation_time(self, friction_path, forest_path, population_path, out_path):
        forest = RasterLayer('rasters', 'forest', layer_path=forest_path, resample='mode')
        friction = RasterLayer('rasters', 'forest', layer_path=friction_path, resample='average')

        forest.align(population_path, out_path)
        friction.align(population_path, out_path)

        forest.add_friction_raster(friction)
        forest.travel_time(out_path)

        self.travel_time = 2 * forest.distance_raster.layer

    def total_time(self, specs_file, friction_path, forest_path, population_path, out_path):
        self.transportation_time(friction_path, forest_path, population_path, out_path)
        self.total_time_yr = self.time_of_cooking * specs_file['Meals_per_day'] * 365 + (
                    self.travel_time + self.time_of_collection) * 52
