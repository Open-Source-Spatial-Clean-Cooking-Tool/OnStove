import numpy as np
import pandas as pd
import os
import rasterio

from math import exp, ceil

from rasterio.fill import fillnodata

from .layer import VectorLayer, RasterLayer
from .raster import interpolate


class Technology:
    """
    Standard technology class.
    """

    # TODO: check if is_clean is needed
    def __init__(self,
                 name=None,
                 carbon_intensity=None,
                 co2_intensity=0,
                 ch4_intensity=0,
                 n2o_intensity=0,
                 co_intensity=0,
                 bc_intensity=0,
                 oc_intensity=0,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=0,
                 is_base=False,
                 transport_cost=0,
                 is_clean=False,
                 current_share_urban=0,
                 current_share_rural=0):  # 24-h PM2.5 concentration

        self.name = name
        self.carbon_intensity = carbon_intensity
        self.co2_intensity = co2_intensity
        self.ch4_intensity = ch4_intensity
        self.n2o_intensity = n2o_intensity
        self.co_intensity = co_intensity
        self.bc_intensity = bc_intensity
        self.oc_intensity = oc_intensity
        self.energy_content = energy_content
        self.tech_life = tech_life
        self.fuel_cost = fuel_cost
        self.inv_cost = inv_cost
        self.om_cost = om_cost
        self.time_of_cooking = time_of_cooking
        self.efficiency = efficiency
        self.pm25 = pm25
        self.time_of_collection = 0
        self.fuel_use = None
        self.is_base = is_base
        self.transport_cost = transport_cost
        self.carbon = None
        self.total_time_yr = None
        self.is_clean = is_clean
        self.current_share_urban = current_share_urban
        self.current_share_rural = current_share_rural
        self.energy = 0
        for paf in ['paf_alri_', 'paf_copd_', 'paf_ihd_', 'paf_lc_', 'paf_stroke_']:
            for s in ['u', 'r']:
                self[paf + s] = 0
        self.discounted_fuel_cost = 0

    def __setitem__(self, idx, value):
        if idx == 'name':
            self.name = value
        elif idx == 'energy_content':
            self.energy_content = value
        elif idx == 'carbon_intensity':
            self.carbon_intensity = value
        elif idx.lower() == 'co2_intensity':
            self.co2_intensity = value
        elif idx.lower() == 'ch4_intensity':
            self.ch4_intensity = value
        elif idx.lower() == 'n2o_intensity':
            self.n2o_intensity = value
        elif idx.lower() == 'co_intensity':
            self.co_intensity = value
        elif idx.lower() == 'bc_intensity':
            self.bc_intensity = value
        elif idx.lower() == 'oc_intensity':
            self.oc_intensity = value
        elif idx == 'fuel_cost':
            self.fuel_cost = value
        elif idx == 'tech_life':
            self.tech_life = value
        elif idx == 'inv_cost':
            self.inv_cost = value
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
        elif idx == 'diesel_cost':
            self.diesel_cost = value
        elif idx == 'carbon':
            self.carbon = value
        elif idx == 'current_share_urban':
            self.current_share_urban = value
        elif idx == 'current_share_rural':
            self.current_share_rural = value
        elif idx == 'paf_alri_r':
            self.paf_alri_r = value
        elif idx == 'paf_copd_r':
            self.paf_copd_r = value
        elif idx == 'paf_ihd_r':
            self.paf_ihd_r = value
        elif idx == 'paf_lc_r':
            self.paf_lc_r = value
        elif idx == 'paf_stroke_r':
            self.paf_stroke_r = value
        elif idx == 'paf_alri_u':
            self.paf_alri_u = value
        elif idx == 'paf_copd_u':
            self.paf_copd_u = value
        elif idx == 'paf_ihd_u':
            self.paf_ihd_u = value
        elif idx == 'paf_lc_u':
            self.paf_lc_u = value
        elif idx == 'paf_stroke_u':
            self.paf_stroke_u = value
        else:
            raise KeyError(idx)

    def __getitem__(self, idx):
        if idx == 'name':
            return self.name
        elif idx == 'energy_content':
            return self.energy_content
        elif idx == 'carbon_intensity':
            return self.carbon_intensity
        elif idx.lower() == 'co2_intensity':
            return self.co2_intensity
        elif idx.lower() == 'ch4_intensity':
            return self.ch4_intensity
        elif idx.lower() == 'n2o_intensity':
            return self.n2o_intensity
        elif idx.lower() == 'co_intensity':
            return self.co_intensity
        elif idx.lower() == 'bc_intensity':
            return self.bc_intensity
        elif idx.lower() == 'oc_intensity':
            return self.oc_intensity
        elif idx == 'fuel_cost':
            return self.fuel_cost
        elif idx == 'tech_life':
            return self.tech_life
        elif idx == 'inv_cost':
            return self.inv_cost
        elif idx == 'om_cost':
            return self.om_cost
        elif idx == 'time_of_cooking':
            return self.time_of_cooking
        elif idx == 'efficiency':
            return self.efficiency
        elif idx == 'pm25':
            return self.pm25
        elif idx == 'time_of_collection':
            return self.time_of_collection
        elif idx == 'fuel_use':
            return self.fuel_use
        elif idx == 'is_base':
            return self.is_base
        elif idx == 'diesel_cost':
            return self.diesel_cost
        elif idx == 'carbon':
            return self.carbon
        elif idx == 'current_share_urban':
            return self.current_share_urban
        elif idx == 'current_share_rural':
            return self.current_share_rural
        elif idx == 'paf_alri_r':
            return self.paf_alri_r
        elif idx == 'paf_copd_r':
            return self.paf_copd_r
        elif idx == 'paf_ihd_r':
            return self.paf_ihd_r
        elif idx == 'paf_lc_r':
            return self.paf_lc_r
        elif idx == 'paf_stroke_r':
            return self.paf_stroke_r
        elif idx == 'paf_alri_u':
            return self.paf_alri_u
        elif idx == 'paf_copd_u':
            return self.paf_copd_u
        elif idx == 'paf_ihd_u':
            return self.paf_ihd_u
        elif idx == 'paf_lc_u':
            return self.paf_lc_u
        elif idx == 'paf_stroke_u':
            return self.paf_stroke_u
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

        if self.pm25 < 7.359:
            rr_stroke = 1
        else:
            rr_stroke = 1 + 1.312 * (1 - exp(-0.012 * (self.pm25 - 7.359) ** 1.273))

        return rr_alri, rr_copd, rr_ihd, rr_lc, rr_stroke

    def paf(self, rr, sfu):

        paf = (sfu * (rr - 1)) / (sfu * (rr - 1) + 1)

        return paf

    @staticmethod
    def discount_factor(specs):
        """
        :param model: onsstove instance containing the informtion of the model
        :return: discount factor to be used for all costs in the net benefit fucntion and the years of analysis
        """
        if specs["Start_year"] == specs["End_year"]:
            proj_life = 1
        else:
            proj_life = specs["End_year"] - specs["Start_year"]

        year = np.arange(proj_life) + 1

        discount_factor = (1 + specs["Discount_rate_social"]) ** year

        return discount_factor, proj_life

    def required_energy(self, model):
        self.energy = model.specs["Meals_per_day"] * 365 * model.energy_per_meal / self.efficiency
        # discount_rate, proj_life = self.discount_factor(specs_file)
        # energy_needed = self.energy * np.ones(proj_life)
        # self.discounted_energy = (energy_needed / discount_rate)

    def get_carbon_intensity(self, model):
        pollutants = ['co2', 'ch4', 'n2o', 'co', 'bc', 'oc']
        self.carbon_intensity = sum([self[f'{pollutant}_intensity'] * model.gwp[pollutant] for pollutant in pollutants])

    def carb(self, model):
        self.required_energy(model)
        if self.carbon_intensity is None:
            self.get_carbon_intensity(model)
        self.carbon = pd.Series([(self.energy * self.carbon_intensity) / 1000] * model.gdf.shape[0],
                                index=model.gdf.index)

    def carbon_emissions(self, model):
        self.carb(model)
        proj_life = model.specs['End_year'] - model.specs['Start_year']
        carbon = model.specs["Cost of carbon emissions"] * (model.base_fuel.carbon - self.carbon) / 1000 / (
                1 + model.specs["Discount_rate_social"]) ** (proj_life)

        self.decreased_carbon_emissions = model.base_fuel.carbon - self.carbon
        self.decreased_carbon_costs = carbon

    def health_parameters(self, model):
        rr_alri, rr_copd, rr_ihd, rr_lc, rr_stroke = self.relative_risk()
        self.paf_alri_r = self.paf(rr_alri, 1 - model.clean_cooking_access_r)
        self.paf_copd_r = self.paf(rr_copd, 1 - model.clean_cooking_access_r)
        self.paf_ihd_r = self.paf(rr_ihd, 1 - model.clean_cooking_access_r)
        self.paf_lc_r = self.paf(rr_lc, 1 - model.clean_cooking_access_r)
        self.paf_stroke_r = self.paf(rr_stroke, 1 - model.clean_cooking_access_r)

        self.paf_alri_u = self.paf(rr_alri, 1 - model.clean_cooking_access_u)
        self.paf_copd_u = self.paf(rr_copd, 1 - model.clean_cooking_access_u)
        self.paf_ihd_u = self.paf(rr_ihd, 1 - model.clean_cooking_access_u)
        self.paf_lc_u = self.paf(rr_lc, 1 - model.clean_cooking_access_u)
        self.paf_stroke_u = self.paf(rr_stroke, 1 - model.clean_cooking_access_u)

    def mortality(self, model):
        """
        Calculates mortality rate per fuel

        Returns
        ----------
        Monetary mortality for each stove in urban and rural settings
        """
        self.health_parameters(model)

        mort_alri_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_alri_u - self.paf_alri_u) * (model.specs["Mort_ALRI"] / 100000)
        mort_copd_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_copd_u - self.paf_copd_u) * (model.specs["Mort_COPD"] / 100000)
        mort_ihd_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_ihd_u - self.paf_ihd_u) * (model.specs["Mort_IHD"] / 100000)
        mort_lc_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_lc_u - self.paf_lc_u) * (model.specs["Mort_LC"] / 100000)
        mort_stroke_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_stroke_u - self.paf_stroke_u) * (model.specs["Mort_STROKE"] / 100000)

        mort_alri_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_alri_r - self.paf_alri_r) * (model.specs["Mort_ALRI"] / 100000)
        mort_copd_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_copd_r - self.paf_copd_r) * (model.specs["Mort_COPD"] / 100000)
        mort_ihd_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_ihd_r - self.paf_ihd_r) * (model.specs["Mort_IHD"] / 100000)
        mort_lc_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_lc_r - self.paf_lc_r) * (model.specs["Mort_LC"] / 100000)
        mort_stroke_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_stroke_r - self.paf_stroke_r) * (model.specs["Mort_STROKE"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_stroke = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        mort_vector_u = []
        mort_vector_r = []

        while i < 6:
            mortality_alri_u = cl_alri[i] * model.specs["VSL"] * mort_alri_u / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            mortality_copd_u = cl_copd[i] * model.specs["VSL"] * mort_copd_u / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_lc_u = cl_lc[i] * model.specs["VSL"] * mort_lc_u / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_ihd_u = cl_ihd[i] * model.specs["VSL"] * mort_ihd_u / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_stroke_u = cl_stroke[i] * model.specs["VSL"] * mort_stroke_u / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)

            mort_total_u = mortality_alri_u + mortality_copd_u + mortality_lc_u + mortality_ihd_u + mortality_stroke_u

            mort_vector_u.append(mort_total_u)

            mortality_alri_r = cl_alri[i] * model.specs["VSL"] * mort_alri_r / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            mortality_copd_r = cl_copd[i] * model.specs["VSL"] * mort_copd_r / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_lc_r = cl_lc[i] * model.specs["VSL"] * mort_lc_r / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_ihd_r = cl_ihd[i] * model.specs["VSL"] * mort_ihd_r / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)
            mortality_stroke_r = cl_stroke[i] * model.specs["VSL"] * mort_stroke_r / (
                    1 + model.specs["Discount_rate_social"]) ** (i - 1)

            mort_total_r = mortality_alri_r + mortality_copd_r + mortality_lc_r + mortality_ihd_r + mortality_stroke_r

            mort_vector_r.append(mort_total_r)

            i += 1

        mortality_u = np.sum(mort_vector_u)
        mortality_r = np.sum(mort_vector_r)

        #  Distributed mortality per household
        distributed_mortality = pd.Series(index=model.gdf.index)

        distributed_mortality[model.gdf["IsUrban"] > 20] = model.gdf.loc[
                                                               model.gdf["IsUrban"] > 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, 'Households']) * mortality_u

        distributed_mortality[model.gdf["IsUrban"] < 20] = model.gdf.loc[
                                                               model.gdf["IsUrban"] < 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, 'Households']) * mortality_r

        self.distributed_mortality = distributed_mortality
        #  Total deaths avoided

        deaths_avoided = pd.Series(index=model.gdf.index)
        deaths_avoided[model.gdf["IsUrban"] > 20] = (
                                                            mort_alri_u + mort_copd_u + mort_lc_u + mort_ihd_u + mort_stroke_u) * (
                                                            model.gdf.loc[
                                                                model.gdf["IsUrban"] > 20, "Calibrated_pop"] / (
                                                                    model.gdf.loc[model.gdf[
                                                                                      "IsUrban"] > 20, "Calibrated_pop"].sum() *
                                                                    model.gdf.loc[model.gdf[
                                                                                      "IsUrban"] > 20, 'Households']))

        deaths_avoided[model.gdf["IsUrban"] < 20] = (
                                                            mort_alri_r + mort_copd_r + mort_lc_r + mort_ihd_r + mort_stroke_r) * (
                                                            model.gdf.loc[
                                                                model.gdf["IsUrban"] < 20, "Calibrated_pop"] / (
                                                                    model.gdf.loc[model.gdf[
                                                                                      "IsUrban"] < 20, "Calibrated_pop"].sum() *
                                                                    model.gdf.loc[model.gdf[
                                                                                      "IsUrban"] < 20, 'Households']))

        self.deaths_avoided = deaths_avoided

    def morbidity(self, model):
        """
        Calculates morbidity rate per fuel

        Returns
        ----------
        Monetary morbidity for each stove in urban and rural settings
        """
        self.health_parameters(model)

        morb_alri_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_alri_u - self.paf_alri_u) * (model.specs["Morb_ALRI"] / 100000)
        morb_copd_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_copd_u - self.paf_copd_u) * (model.specs["Morb_COPD"] / 100000)
        morb_ihd_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_ihd_u - self.paf_ihd_u) * (model.specs["Morb_IHD"] / 100000)
        morb_lc_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_lc_u - self.paf_lc_u) * (model.specs["Morb_LC"] / 100000)
        morb_stroke_u = model.gdf.loc[model.gdf["IsUrban"] > 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_stroke_u - self.paf_stroke_u) * (model.specs["Morb_STROKE"] / 100000)

        morb_alri_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_alri_r - self.paf_alri_r) * (model.specs["Morb_ALRI"] / 100000)
        morb_copd_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_copd_r - self.paf_copd_r) * (model.specs["Morb_COPD"] / 100000)
        morb_ihd_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_ihd_r - self.paf_ihd_r) * (model.specs["Morb_IHD"] / 100000)
        morb_lc_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_lc_r - self.paf_lc_r) * (model.specs["Morb_LC"] / 100000)
        morb_stroke_r = model.gdf.loc[model.gdf["IsUrban"] < 20, "Calibrated_pop"].sum() * (
                model.base_fuel.paf_stroke_r - self.paf_stroke_r) * (model.specs["Morb_STROKE"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_stroke = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        morb_vector_u = []
        morb_vector_r = []

        while i < 6:
            morbidity_alri_u = cl_alri[i] * model.specs["COI_ALRI"] * morb_alri_u / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                       i - 1)
            morbidity_copd_u = cl_copd[i] * model.specs["COI_COPD"] * morb_copd_u / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                       i - 1)
            morbidity_lc_u = cl_lc[i] * model.specs["COI_LC"] * morb_lc_u / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            morbidity_ihd_u = cl_ihd[i] * model.specs["COI_IHD"] * morb_ihd_u / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            morbidity_stroke_u = cl_stroke[i] * model.specs["COI_STROKE"] * morb_stroke_u / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                         i - 1)

            morb_total_u = morbidity_alri_u + morbidity_copd_u + morbidity_lc_u + morbidity_ihd_u + morbidity_stroke_u

            morb_vector_u.append(morb_total_u)

            morbidity_alri_r = cl_alri[i] * model.specs["COI_ALRI"] * morb_alri_r / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                       i - 1)
            morbidity_copd_r = cl_copd[i] * model.specs["COI_COPD"] * morb_copd_r / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                       i - 1)
            morbidity_lc_r = cl_lc[i] * model.specs["COI_LC"] * morb_lc_r / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            morbidity_ihd_r = cl_ihd[i] * model.specs["COI_IHD"] * morb_ihd_r / (1 + model.specs["Discount_rate_social"]) ** (
                    i - 1)
            morbidity_stroke_r = cl_stroke[i] * model.specs["COI_STROKE"] * morb_stroke_r / (
                    1 + model.specs["Discount_rate_social"]) ** (
                                         i - 1)

            morb_total_r = morbidity_alri_r + morbidity_copd_r + morbidity_lc_r + morbidity_ihd_r + morbidity_stroke_r

            morb_vector_r.append(morb_total_r)

            i += 1

        morbidity_u = np.sum(morb_vector_u)
        morbidity_r = np.sum(morb_vector_r)

        distributed_morbidity = pd.Series(index=model.gdf.index)

        distributed_morbidity[model.gdf["IsUrban"] > 20] = model.gdf.loc[
                                                               model.gdf["IsUrban"] > 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, 'Households']) * morbidity_u

        distributed_morbidity[model.gdf["IsUrban"] < 20] = model.gdf.loc[
                                                               model.gdf["IsUrban"] < 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, 'Households']) * morbidity_r

        self.distributed_morbidity = distributed_morbidity

        cases_avoided = pd.Series(index=model.gdf.index)
        cases_avoided[model.gdf["IsUrban"] > 20] = (
                                                           morb_alri_u + morb_copd_u + morb_lc_u + morb_ihd_u + morb_stroke_u) * (
                                                           model.gdf.loc[
                                                               model.gdf["IsUrban"] > 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] > 20, 'Households']))

        cases_avoided[model.gdf["IsUrban"] < 20] = (
                                                           morb_alri_r + morb_copd_r + morb_lc_r + morb_ihd_r + morb_stroke_r) * (
                                                           model.gdf.loc[
                                                               model.gdf["IsUrban"] < 20, "Calibrated_pop"] / (
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, "Calibrated_pop"].sum() *
                                                                   model.gdf.loc[model.gdf[
                                                                                     "IsUrban"] < 20, 'Households']))

        self.cases_avoided = cases_avoided

    def salvage(self, model):
        """
        Calculates discounted salvage cost assuming straight-line depreciation
        Returns
        ----------
        discounted salvage cost
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        salvage = np.zeros(proj_life)
        used_life = proj_life % self.tech_life
        used_life_base = proj_life % model.base_fuel.tech_life

        base_salvage = model.base_fuel.inv_cost * (1 - used_life_base / model.base_fuel.tech_life)
        salvage[-1] = self.inv_cost * (1 - used_life / self.tech_life)

        salvage = salvage.sum() - base_salvage
        # TODO: this needs to be changed to use a series for each salvage value
        discounted_salvage = salvage / discount_rate

        self.discounted_salvage_cost = discounted_salvage

    def discounted_om(self, model):
        """
        Calls discount_factor function and creates discounted OM costs.
        Returns
        ----------
        discountedOM costs for each stove during the project lifetime
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        operation_and_maintenance = self.om_cost * np.ones(proj_life) #* self.inv_cost
        # operation_and_maintenance[0] = 0

        # i = self.tech_life
        # while i < proj_life:
        #     operation_and_maintenance[i] = 0
        #     i = i + self.tech_life
        # TODO: this needs to be changed to use a series for each om value adn relative to the base fuel

        # self.discounted_om_costs = discounted_om_cost

        discounted_om = np.array([sum((operation_and_maintenance - x) / discount_rate) for
                                  x in model.base_fuel.om_cost])
        self.discounted_om_costs = pd.Series(discounted_om, index=model.gdf.index)

    def discounted_inv(self, model):
        """
        Calls discount_factor function and creates discounted investment cost. Uses proj_life and tech_life to determine
        number of necessary re-investments

        Returns
        ----------
        discounted investment cost for each stove during the project lifetime
        """
        discount_rate, proj_life = self.discount_factor(model.specs)

        investments = np.zeros(proj_life)
        # investments[0] = self.inv_cost

        i = self.tech_life
        while i < proj_life:
            investments[i] = self.inv_cost
            i = i + self.tech_life

        base_investments = np.zeros(model.base_fuel.inv_cost.shape[0])
        j = 0
        # TODO: make sure the shapes of the base_fuel and the tech are consistent
        for cost, life in zip(model.base_fuel.inv_cost, model.base_fuel.tech_life):
            _base_investments = np.zeros(proj_life)
            i = ceil(life)
            while i < proj_life:
                _base_investments[i] = cost
                base_investments[j] = _base_investments
                i = i + ceil(life)
                j += 1

        # discounted_investments = (model.base_fuel.inv_cost - investments) / discount_rate
        investments_discounted = np.array([sum((investments - x) / discount_rate) for x in base_investments])
        self.discounted_investments = pd.Series(investments_discounted, index=model.gdf.index) + (
                self.inv_cost - model.base_fuel.inv_cost)

    def discount_fuel_cost(self, model, relative=True):
        self.required_energy(model)
        discount_rate, proj_life = self.discount_factor(model.specs)

        # base_cost = (
        #                     model.base_fuel.energy * model.base_fuel.fuel_cost / model.base_fuel.energy_content + model.base_fuel.transport_cost) * np.ones(
        #     model.gdf.shape[0])

        cost = (self.energy * self.fuel_cost / self.energy_content + self.transport_cost) * np.ones(
            model.gdf.shape[0]) #- base_cost

        fuel_cost = [np.ones(proj_life) * x for x in cost]

        fuel_cost_discounted = np.array([sum(x / discount_rate) for x in fuel_cost])

        if relative:
            discounted_base_fuel_cost = model.base_fuel.discounted_fuel_cost
        else:
            discounted_base_fuel_cost = 0

        self.discounted_fuel_cost = pd.Series(fuel_cost_discounted, index=model.gdf.index) - discounted_base_fuel_cost

    def total_time(self, model):
        self.total_time_yr = (self.time_of_cooking + self.time_of_collection) * 365

    def time_saved(self, model):
        # if self.is_base:
        #     self.total_time_saved = np.zeros(model.gdf.shape[0])
        #     self.time_value = np.zeros(model.gdf.shape[0])
        # else:
        proj_life = model.specs['End_year'] - model.specs['Start_year']
        self.total_time(model)
        # self.total_time_saved = model.base_fuel.total_time_yr - self.total_time_yr  # time saved per household
        self.total_time_saved = model.base_fuel.total_time_yr - self.total_time_yr
        # time value of time saved per sq km
        self.time_value = self.total_time_saved * model.gdf["value_of_time"] / (
                1 + model.specs["Discount_rate_social"]) ** (proj_life)

    def total_costs(self):
        self.costs = (self.discounted_fuel_cost + self.discounted_investments +  # - self.time_value +
                      self.discounted_om_costs - self.discounted_salvage_cost)

    def net_benefit(self, model, w_health=1, w_environment=1, w_social=1, w_costs=1):
        self.total_costs()
        self.benefits = w_health * (
                self.distributed_morbidity + self.distributed_mortality) + w_environment * self.decreased_carbon_costs + w_social * self.time_value
        model.gdf["costs_{}".format(self.name)] = self.costs
        model.gdf["benefits_{}".format(self.name)] = self.benefits
        model.gdf["net_benefit_{}".format(self.name)] = self.benefits - w_costs * self.costs
        self.factor = pd.Series(np.ones(model.gdf.shape[0]), index=model.gdf.index)
        self.households = model.gdf['Households']


class LPG(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=None,  # Kg/GJ
                 co2_intensity=63,
                 ch4_intensity=0.003,
                 n2o_intensity=0.0001,
                 co_intensity=0,
                 bc_intensity=0.0044,
                 oc_intensity=0.0091,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=43,
                 travel_time=None,
                 truck_capacity=2000,
                 diesel_cost=0.88,
                 diesel_per_hour=14,
                 lpg_path=None,
                 friction_path=None):
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life, inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=True)
        self.travel_time = travel_time
        self.truck_capacity = truck_capacity
        self.diesel_cost = diesel_cost
        self.diesel_per_hour = diesel_per_hour
        self.transport_cost = None
        self.lpg_path = lpg_path
        self.friction_path = friction_path

    def add_travel_time(self, model, align=False):
        lpg = VectorLayer(self.name, 'Suppliers', layer_path=self.lpg_path)
        friction = RasterLayer(self.name, 'Friction', layer_path=self.friction_path, resample='average')

        if align:
            os.makedirs(os.path.join(model.output_directory, self.name, 'Suppliers'), exist_ok=True)
            lpg.reproject(model.base_layer.meta['crs'], os.path.join(model.output_directory, self.name, 'Suppliers'))
            friction.align(model.base_layer.path, os.path.join(model.output_directory, self.name, 'Friction'))

        lpg.add_friction_raster(friction)
        lpg.travel_time(os.path.join(model.output_directory, self.name))
        interpolate(lpg.distance_raster.path)
        self.travel_time = 2 * lpg.distance_raster.layer

    def transportation_cost(self, model):
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
        transport_cost = (self.diesel_per_hour * self.diesel_cost * self.travel_time) / self.truck_capacity
        kg_yr = (model.specs["Meals_per_day"] * 365 * model.energy_per_meal) / (
                self.efficiency * self.energy_content)  # energy content in MJ/kg
        transport_cost = transport_cost * kg_yr
        transport_cost[transport_cost < 0] = np.nan
        self.transport_cost = model.raster_to_dataframe(transport_cost, nodata=np.nan,
                                                        fill_nodata='interpolate', method='read')

    def discount_fuel_cost(self, model, relative=True):
        self.transportation_cost(model)
        super().discount_fuel_cost(model, relative)

    def transport_emissions(self, model):
        # Diesel consumption per h is assumed to be 14 l/h (14 l/100km)
        # Carbon intensity from https://www.eia.gov/environment/emissions/co2_vol_mass.php
        # is 10.19 kg/gallon, converted to liters is 2.69 kg/liter
        kg_yr = (model.specs["Meals_per_day"] * 365 * model.energy_per_meal) / (
                self.efficiency * self.energy_content)
        diesel_consumption = self.travel_time * 14
        hh_emissions = 2.69 * diesel_consumption / self.truck_capacity * kg_yr  # kgCO2/l * l/trip * trip/kgLPG * kgLPG/yr
        return model.raster_to_dataframe(hh_emissions, nodata=np.nan,
                                         fill_nodata='interpolate', method='read')

    def carb(self, model):
        super().carb(model)
        self.carbon += self.transport_emissions(model)


class Biomass(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=None,
                 co2_intensity=112,
                 ch4_intensity=0.864,
                 n2o_intensity=0.0039,
                 co_intensity=0,
                 bc_intensity=0.1075,
                 oc_intensity=0.308,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=844,
                 forest_path=None,
                 friction_path=None,
                 travel_time=None,
                 collection_capacity=25):
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=False)
        self.forest_condition = None
        self.travel_time = travel_time
        self.forest_path = forest_path
        self.friction_path = friction_path
        self.collection_capacity = collection_capacity

    def transportation_time(self, friction_path, forest_path, model, align=False):
        forest = RasterLayer(self.name, 'Forest', layer_path=forest_path, resample='mode')
        friction = RasterLayer(self.name, 'Friction', layer_path=friction_path, resample='average')

        if align:
            forest.align(model.base_layer.path, os.path.join(model.output_directory, self.name, 'Forest'))
            friction.align(model.base_layer.path, os.path.join(model.output_directory, self.name, 'Friction'))

        forest.add_friction_raster(friction)
        forest.travel_time(condition=self.forest_condition)

        self.travel_time = 2 * model.raster_to_dataframe(forest.distance_raster.layer,
                                                         nodata=forest.distance_raster.meta['nodata'],
                                                         fill_nodata='interpolate', method='read')

    def total_time(self, model):
        self.transportation_time(self.friction_path, self.forest_path, model)
        trips_per_yr = self.energy / (self.collection_capacity * self.energy_content)
        self.total_time_yr = self.time_of_cooking * model.specs['Meals_per_day'] * 365 + (
                self.travel_time + self.time_of_collection) * trips_per_yr

    def get_carbon_intensity(self, model):
        intensity = self['co2_intensity']
        self['co2_intensity'] *= model.specs['fnrb']
        super().get_carbon_intensity(model)
        self['co2_intensity'] = intensity


class Charcoal(Technology):
    def get_carbon_intensity(self, model):
        intensity = self['co2_intensity']
        self['co2_intensity'] *= model.specs['fnrb']
        super().get_carbon_intensity(model)
        self['co2_intensity'] = intensity

class Electricity(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=None,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 connection_cost=0,  # cost of additional infrastructure
                 grid_capacity_cost=None,
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=41):
        super().__init__(name, carbon_intensity, None, None, None,
                         None, None, None, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=True)
        # Carbon intensity of fossil fuel plants in kg/GWh
        self.generation = {}

        self.capacity = {}
        self.grid_capacity_cost = grid_capacity_cost

        self.tiers_path = None
        self.connection_cost = connection_cost
        self.carbon_intensities = {'coal': 0.090374363, 'natural_gas': 0.050300655,
                                   'crude_oil': 0.070650288, 'heavy_fuel_oil': 0.074687989,
                                   'oil': 0.072669139, 'diesel': 0.069332823,
                                   'still_gas': 0.060849859, 'flared_natural_gas': 0.051855075,
                                   'waste': 0.010736111, 'biofuels_and_waste': 0.010736111,
                                   'nuclear': 0, 'hydro': 0, 'wind': 0,
                                   'solar': 0, 'other': 0, 'geothermal': 0}

        # TODO: make this general, with other fuel mix this crash
        self.grid_capacity_costs = {'oil': 1467, 'natural_gas': 550,
                                    'biofuels_and_waste': 2117,
                                    'nuclear': 4000, 'hydro': 2100, 'coal': 1600, 'wind': 1925,
                                    'solar': 1400, 'geothermal': 2917}

        self.grid_techs_life = {'oil': 35, 'natural_gas': 30,
                                'biofuels_and_waste': 25,
                                'nuclear': 50, 'hydro': 60, 'coal': 40, 'wind': 22,
                                'solar': 25, 'geothermal': 30}

    def __setitem__(self, idx, value):
        if 'generation' in idx:
            self.generation[idx.lower().replace('generation_', '')] = value
        elif 'grid_capacity_cost' in idx:
            self.grid_capacity_cost = value
        elif 'capacity' in idx:
            self.capacity[idx.lower().replace('capacity_', '')] = value
        elif 'carbon_intensity' == idx:
            self.carbon_intensity = value
        elif 'carbon_intensity' in idx:
            self.carbon_intensities[idx.lower().replace('carbon_intensity_', '')] = value
        elif 'connection_cost' in idx:
            self.connection_cost = value
        elif 'grid_cap_life' in idx:
            self.grid_cap_life = value
        else:
            super().__setitem__(idx, value)

    def get_capacity_cost(self, model):
        # TODO: this line assumes if no tiers data is added, that all population settlements will need added capacity
        self.required_energy(model)
        if self.tiers_path is None:
            add_capacity = 1
        else:
            model.raster_to_dataframe(self.tiers_path, name='Electricity_tiers', method='sample')
            self.tiers = model.gdf['Electricity_tiers'].copy()
            add_capacity = (self.tiers < 3)

        if self.grid_capacity_cost is None:
            self.get_grid_capacity_cost()
            salvage = self.grid_salvage(model)
        else:
            salvage = self.grid_salvage(model, True)

        self.capacity = self.energy * add_capacity / (3.6 * self.time_of_cooking * 365)
        self.capacity_cost = self.capacity * (self.grid_capacity_cost - salvage)

    def get_carbon_intensity(self, model):
        grid_emissions = sum([gen * self.carbon_intensities[fuel] for fuel, gen in self.generation.items()])
        grid_generation = sum(self.generation.values())
        self.carbon_intensity = grid_emissions / grid_generation * 1000  # to convert from Mton/PJ to kg/GJ

    # def carb(self, model):
    #     self.get_carbon_intensity()
    #     super().carb(model)

    def get_grid_capacity_cost(self):
        self.grid_capacity_cost = sum(
            [self.grid_capacity_costs[fuel] * (cap / sum(self.capacity.values())) for fuel, cap in
             self.capacity.items()])

    def grid_salvage(self, model, single=False):
        discount_rate, proj_life = self.discount_factor(model.specs)
        if single:
            used_life = proj_life % self.grid_cap_life
            salvage = self.grid_capacity_cost * (1 - used_life / self.grid_cap_life)
        else:
            salvage_values = []

            for tech, cap in self.capacity.items():
                used_life = proj_life % self.grid_techs_life[tech]
                salvage = self.grid_capacity_costs[tech] * (1 - used_life / self.grid_techs_life[tech])
                salvage_values.append(salvage * cap / sum(self.capacity.values()))

            salvage = sum(salvage_values)

        return salvage / discount_rate[0]

    def carb(self, model):
        if self.carbon_intensity is None:
            self.get_carbon_intensity(model)
        super().carb(model)

    def discounted_inv(self, model):
        super().discounted_inv(model)
        self.discounted_investments += self.connection_cost

    def total_costs(self):
        super().total_costs()
        self.costs += self.capacity_cost

    def net_benefit(self, model, w_health=1, w_environment=1, w_social=1, w_costs=1):
        super().net_benefit(model, w_health, w_environment, w_social, w_costs)
        model.gdf.loc[model.gdf['Current_elec'] == 0, "net_benefit_{}".format(self.name)] = np.nan
        # model.gdf.loc[model.gdf['Current_elec'] == 0, "costs_{}".format(self.name)] = np.nan
        # model.gdf.loc[model.gdf['Current_elec'] == 0, "benefits_{}".format(self.name)] = np.nan
        factor = model.gdf['Elec_pop_calib'] / model.gdf['Calibrated_pop']
        factor[factor > 1] = 1
        self.factor = factor
        self.households = model.gdf['Households'] * factor


class Biogas(Technology):
    """
    LPG technology class. Inherits all functionality from the standard
    Technology class
    """

    def __init__(self,
                 name=None,
                 carbon_intensity=None,
                 co2_intensity=0,
                 ch4_intensity=0.0288,
                 n2o_intensity=0.0006,
                 co_intensity=0,
                 bc_intensity=0.0043,
                 oc_intensity=0.0091,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=43,
                 utilization_factor=0.5,
                 digestor_eff=0.4,
                 friction_path=None):
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=True)
        # TODO: Check what's the difference between these two factors
        self.utilization_factor = utilization_factor
        self.digestor_eff = digestor_eff
        self.friction_path = friction_path
        self.water = None
        self.temperature = None

    def read_friction(self, model, friction_path):
        friction = RasterLayer(self.name, 'Friction', layer_path=friction_path, resample='average')
        data = model.raster_to_dataframe(friction.layer, nodata=friction.meta['nodata'],
                                         fill_nodata='interpolate', method='read')
        return data / 60

    def required_energy_hh(self, model):
        self.required_energy(model)
        return self.energy / self.digestor_eff

    def get_collection_time(self, model):
        self.available_biogas(model)
        required_energy_hh = self.required_energy_hh(model)

        friction = self.read_friction(model, self.friction_path)
        time_of_collection = required_energy_hh / (model.gdf["biogas_energy"] / (1000000 * 0.2))
        time_of_collection *= friction
        time_of_collection[time_of_collection > 10] = 10
        self.time_of_collection = time_of_collection


    def available_biogas(self, model):
        # Biogas production potential in liters per day
        from_cattle = model.gdf["Cattles"] * 12 * 0.15 * 0.8 * 305
        from_buffalo = model.gdf["Buffaloes"] * 14 * 0.2 * 0.75 * 305
        from_sheep = model.gdf["Sheeps"] * 0.7 * 0.25 * 0.8 * 452
        from_goat = model.gdf["Goats"] * 0.6 * 0.3 * 0.85 * 450
        from_pig = model.gdf["Pigs"] * 5 * 0.75 * 0.14 * 470
        from_poultry = model.gdf["Poultry"] * 0.12 * 0.25 * 0.75 * 450

        # fraction = self.read_friction(model, self.friction_path) / (1000000 * 0.2)
        # self.fraction = fraction

        model.gdf["available_biogas"] = ((from_cattle + from_buffalo + from_goat + from_pig + from_poultry + \
                                          from_sheep) * self.digestor_eff / 1000) * 365

        # model.gdf["m3_biogas_hh"] = fraction * model.gdf["available_biogas"]

        if self.temperature is not None:
            if isinstance(self.temperature, str):
                self.temperature = RasterLayer('Biogas', 'Temperature', self.temperature)

            model.raster_to_dataframe(self.temperature.layer, name="Temperature", method='read',
                                      nodata=self.temperature.meta['nodata'], fill_nodata='interpolate')
            model.gdf.loc[model.gdf["Temperature"] < 10, "available_biogas"] = 0
            model.gdf.loc[(model.gdf["IsUrban"] > 20), "available_biogas"] = 0

        if self.water is not None:
            if isinstance(self.water, str):
                self.water = VectorLayer('Biogas', 'Water scarcity', self.water, bbox=model.mask_layer.layer)
            model.raster_to_dataframe(self.water.layer, name="Water",
                                      fill_nodata='interpolate', method='read')
            model.gdf.loc[model.gdf["Water"] == 0, "available_biogas"] = 0

        model.gdf["biogas_energy"] = model.gdf["available_biogas"] * self.energy_content

    def recalibrate_livestock(self, model, buffaloes, cattles, poultry, goats, pigs, sheeps):
        paths = {
            'Buffaloes': buffaloes,
            'Cattles': cattles,
            'Poultry': poultry,
            'Goats': goats,
            'Pigs': pigs,
            'Sheeps': sheeps}

        for name, path in paths.items():
            layer = RasterLayer('Livestock', name,
                                layer_path=path)
            model.raster_to_dataframe(layer.layer, name=name, method='read',
                                      nodata=layer.meta['nodata'], fill_nodata='interpolate')

    def total_time(self, model):
        self.get_collection_time(model)
        super().total_time(model)

    def net_benefit(self, model, w_health=1, w_environment=1, w_social=1, w_costs=1):
        super().net_benefit(model, w_health, w_environment, w_social, w_costs)
        # model.gdf.loc[(model.gdf['biogas_energy_hh'] == 0), "benefits_{}".format(self.name)] = np.nan
        required_energy_hh = self.required_energy_hh(model)
        model.gdf.loc[(model.gdf['biogas_energy'] < required_energy_hh), "benefits_{}".format(self.name)] = np.nan
        model.gdf.loc[(model.gdf['biogas_energy'] < required_energy_hh), "net_benefit_{}".format(self.name)] = np.nan
        factor = model.gdf['biogas_energy'] / (required_energy_hh * model.gdf['Households'])
        factor[factor > 1] = 1
        self.factor = factor
        self.households = model.gdf['Households'] * factor

        del model.gdf["Cattles"]
        del model.gdf["Buffaloes"]
        del model.gdf["Sheeps"]
        del model.gdf["Goats"]
        del model.gdf["Pigs"]
        del model.gdf["Poultry"]
