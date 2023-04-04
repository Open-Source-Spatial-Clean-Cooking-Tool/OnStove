"""This module contains the technology classes used in OnStove."""

import os
import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Optional, Callable
from math import exp

from onstove._utils import raster_setter, vector_setter
from onstove.layer import VectorLayer, RasterLayer


class Technology:
    """
    Standard technology class used in order to model the different stoves used in the analysis.

        Parameters
    ----------
    name: str, optional.
        Name of the technology to model.
    carbon_intensity: float, optional
        The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
        gas-specific intensities will be used (e.g. ch4_intensity).
    co2_intensity: float, default 0
        The CO2 emissions in kg/GJ of burned fuel.
    ch4_intensity: float, default 0
        The CH4 emissions in kg/GJ of burned fuel.
    n2o_intensity: float, default 0
        The N2O emissions in kg/GJ of burned fuel.
    co_intensity: float, default 0
        The CO emissions in kg/GJ of burned fuel.
    bc_intensity: float, default 0
        The black carbon emissions in kg/GJ of burned fuel.
    oc_intensity: float, default 0
        The organic carbon emissions in kg/GJ of burned fuel.
    energy_content: float, default 0
        Energy content of the fuel in MJ/kg.
    tech_life: int, default 0
        Stove life in year.
    inv_cost: float, default 0
        Investment cost of the stove in USD.
    fuel_cost: float, default 0
        Fuel cost in USD/kg if any.
    time_of_cooking: float, default 0
        Daily average time spent for cooking with this stove in hours.
    om_cost: float, default 0
        Operation and maintenance cost in USD/year.
    efficiency: float, default 0
        Efficiency of the stove.
    pm25: float, default 0
        Particulate Matter emissions (PM25) in mg/kg of fuel.
    is_base: boolean, default False
        Boolean determining if a specific stove is the base stove for everyone in the area of interest.
    transport_cost: float, default 0
        Cost of transportation
    is_clean: boolean, default False
        Boolean indicating whether the stove is clean or not.
    current_share_urban: float, default 0
        Current share of the stove assessed in the urban areas of the area of interest.
    current_share_rural: float, default 0
        Current share of the stove assessed in the rural areas of the area of interest.
    epsilon: float, default 0.71
        Emissions adjustment factor multiplied with the PM25 emissions.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[str] = None,
                 co2_intensity: float = 0,
                 ch4_intensity: float = 0,
                 n2o_intensity: float = 0,
                 co_intensity: float = 0,
                 bc_intensity: float = 0,
                 oc_intensity: float = 0,
                 energy_content: float = 0,
                 tech_life: int = 0,
                 inv_cost: float = 0,
                 fuel_cost: float = 0,
                 time_of_cooking: float = 0,
                 om_cost: float = 0,
                 efficiency: float = 0,
                 pm25: float = 0,
                 is_base: bool = False,
                 transport_cost: float = 0,
                 is_clean: bool = False,
                 current_share_urban: float = 0,
                 current_share_rural: float = 0,
                 epsilon: float = 0.71):

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
        self.epsilon = epsilon
        for paf in ['paf_alri', 'paf_copd', 'paf_ihd', 'paf_lc', 'paf_stroke']:
            self[paf] = 0
        self.discounted_fuel_cost = 0
        self.factor = None
        self.discounted_investments = 0
        self.discounted_om_costs = 0
        self.discounted_salvage_cost = 0
        self.decreased_carbon_costs = 0
        self.clean_cooking_access = 0
        self.sfu = 0
        self.time_value = 0
        self.costs = None
        self.benefits = None
        self.deaths_avoided = None
        self.cases_avoided = None
        self.pop_sqkm = None
        self.distributed_morbidity = None
        self.distributed_mortality = None
        self.time_value = None
        self.decreased_carbon_emissions = None
        self.net_benefits = None
        self.deaths = None
        self.cases = None

    def __setitem__(self, idx, value):
        self.__dict__[idx] = value

    def __getitem__(self, idx):
        return self.__dict__[idx]

    def adjusted_pm25(self):
        """Adjusts the PM25 value of each stove based on the adjusment factor. This is to take into account the
        potential behaviour change resulting from stove change [1]_.

        References
        ----------
        .. [1] Das, I. et al. The benefits of action to reduce household air pollution (BAR-HAP) model:
            A new decision support tool. PLOS ONE 16, e0245729 (2021).
        """
        self.pm25 *= self.epsilon

    def relative_risk(self) -> float:
        """Calculates the relative risk of contracting ALRI, COPD, IHD, lung cancer or stroke based on the adjusted
          PM25 emissions. The equations and parameters used are based on the work done by Burnett et al.[1]_

        References
        ----------
        .. [1] Burnett, R. T. et al. An Integrated Risk Function for Estimating the Global Burden of Disease
            Attributable to Ambient Fine Particulate Matter Exposure.
            Environmental Health Perspectives 122, 397–403 (2014).

        Returns
        -------
        rr_alri: float
            Relative Risk of ALRI
        rr_copd: float
            Relative Risk of COPD
        rr_ihd: float
            Relative Risk of IHD
        rr_lc: float
            Relative Risk of lung cancer
        rr_stroke: float
            Relative Risk of stroke
        """
        if self.pm25 < 7.298:
            rr_alri = 1
        else:
            rr_alri = 1 + 2.383 * (1 - exp(-0.004 * (self.pm25 - 7.298) ** 1.193))

        if self.pm25 < 7.337:
            rr_copd = 1
        else:
            rr_copd = 1 + 22.485 * (1 - exp(-0.001 * (self.pm25 - 7.337) ** 0.694))

        if self.pm25 < 7.449:
            rr_ihd = 1
        else:
            rr_ihd = 1 + 1.647 * (1 - exp(-0.048 * (self.pm25 - 7.449) ** 0.467))

        if self.pm25 < 7.345:
            rr_lc = 1
        else:
            rr_lc = 1 + 152.496 * (1 - exp(-0.000167 * (self.pm25 - 7.345) ** 0.76))

        if self.pm25 < 7.358:
            rr_stroke = 1
        else:
            rr_stroke = 1 + 1.314 * (1 - exp(-0.012 * (self.pm25 - 7.358) ** 1.275))

        return rr_alri, rr_copd, rr_ihd, rr_lc, rr_stroke

    def paf(self, rr: float, sfu: float) -> float:
        """Calculates the population attributable fraction for ALRI, COPD, IHD, lung cancer or stroke
        based on the percentage of population using non-clean stoves and the relative risk [1]_.

        References
        ----------
        .. [1] Jeuland, M., Tan Soo, J.-S. & Shindell, D. The need for policies to reduce the costs of cleaner
            cooking in low income settings: Implications from systematic analysis of costs and benefits.
            Energy Policy 121, 275–285 (2018).

        Parameters
        ----------
        rr: float
            The relative risk of contracting ALRI, COPD, IHD, lung cancer and stroke.
        sfu: float
            Solid Fuel Users. This is the percentage of people using traditional cooking fuels. This is read from the
            techno-economic specification file as fuels not being clean

        Returns
        -------
        paf: float
            The Population Attributable Fraction for each disease
        """
        paf = (sfu * (rr - 1)) / (sfu * (rr - 1) + 1)
        return paf

    @staticmethod
    def discount_factor(specs: dict) -> tuple[list[float], list[float]]:
        """Calculates and returns the discount factor used for benefits and costs in the net-benefit equation. Also
        returns the length of the analysis in years

        Parameters
        ----------
        specs: dict
            The socio-economic specification file containing socio-economic data applying to your study area

        Returns
        -------
        Discount factor and the project life
        """
        if specs["start_year"] == specs["end_year"]:
            proj_life = 1
        else:
            proj_life = specs["end_year"] - specs["start_year"]

        year = np.arange(proj_life) + 1

        discount_factor = (1 + specs["discount_rate"]) ** year

        return discount_factor, proj_life

    def required_energy(self, model: 'onstove.OnStove'):
        """ Calculates the annual energy needed for cooking in MJ/yr. This is dependent on the number of meals cooked
        and the efficiency of the stove.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """

        self.energy = model.specs["meals_per_day"] * 365 * model.energy_per_meal / self.efficiency

    def get_carbon_intensity(self, model: 'onstove.OnStove'):
        """Calculates the carbon intensity of the associated stove.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        pollutants = ['co2', 'ch4', 'n2o', 'co', 'bc', 'oc']
        self.carbon_intensity = sum([self[f'{pollutant}_intensity'] * model.gwp[pollutant] for pollutant in pollutants])

    def carb(self, model: 'onstove.OnStove', mask: pd.Series):
        """Checks if carbon_emission is given in the socio-economic specification file. If it is given this is read
        directly, otherwise the get_carbon_intensity function is called

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        get_carbon_intensity
        """

        self.required_energy(model)
        if self.carbon_intensity is None:
            self.get_carbon_intensity(model)

        self.carbon = pd.Series((self.energy * self.carbon_intensity) / 1000, index=mask.index)


    def carbon_emissions(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """Calculates the reduced emissions and the costs avoided by reducing these emissions.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        year: int
            Defines which year that is being run.
        mask: pd.Series
            Determines which cells are ran. This is relevant when the start and end years are different

        See also
        --------
        carb
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        self.carb(model, mask)

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        if relative:
            start_year = mask[mask].index

            base_annual_carbon = np.repeat(np.expand_dims(model.base_fuel.carbon, axis=1),
                                           model.specs["end_year"] - model.specs["start_year"], axis=1)

            proj_years[mask, (model.year - model.specs["start_year"] - 1):proj_years.shape[1]] = 1

            carbon = proj_years * np.array(self.carbon)[:, None]
            carbon[mask, 0:(model.year - model.specs["start_year"] - 1)] = base_annual_carbon[mask,
                                                                                 0:model.year - model.specs[
                                                                                     "start_year"] - 1]

            if not isinstance(self.decreased_carbon_emissions, pd.Series):
                self.decreased_carbon_emissions = pd.Series(0, index=mask.index, dtype='float64')
                self.decreased_carbon_costs = pd.Series(0, index=mask.index, dtype='float64')

            # TODO: save this one
            decreased_carbon = base_annual_carbon - carbon


            self.decreased_carbon_emissions.loc[start_year] = decreased_carbon[mask].sum(axis=1)

            discounted_carbon = np.array(
                [sum(x * model.specs["cost_of_carbon_emissions"] / (1000 * discount_rate)) for x in decreased_carbon])

            self.decreased_carbon_costs.loc[start_year] = discounted_carbon[mask]
        else:
            proj_years[:] = 1
            carbon = proj_years * np.array(self.carbon)[:, None]
            discounted_carbon = np.array([sum(x * model.specs["cost_of_carbon_emissions"] / (1000 * discount_rate)) for x in carbon])
            self.discounted_carbon_costs = pd.Series(discounted_carbon, index=mask.index)

    def health_parameters(self, model: 'onstove.OnStove'):
        """Calculates the population attributable fraction for ALRI, COPD, IHD, lung cancer or stroke for urban and
        rural settlements of the area of interest.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        relative_risk, paf
        """
        rr_alri, rr_copd, rr_ihd, rr_lc, rr_stroke = self.relative_risk()
        self.paf_alri = self.paf(rr_alri, model.sfu)
        self.paf_copd = self.paf(rr_copd, model.sfu)
        self.paf_ihd = self.paf(rr_ihd, model.sfu)
        self.paf_lc = self.paf(rr_lc, model.sfu)
        self.paf_stroke = self.paf(rr_stroke, model.sfu)

    def mort_morb(self, model: 'onstove.OnStove', mask: gpd.GeoSeries, parameter: str = 'mort', relative = True) -> tuple[
        float, float]:
        """
        Calculates mortality or morbidity rate per fuel. These two calculations are very similar in nature and are
        therefore combined in one function. In order to indicate if morbidity or mortality should be calculated, the
        `parameter` parameter can be changed (to either `Morb` or `Mort`).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        mask: gpd.GeoSeries
            Determines which cells are ran. This is relevant when the start and end years are different
        parameter: str, default 'Mort'
            Parameter to calculate. For mortality enter 'Mort' and for morbidity enter 'Morb'

        Returns
        ----------
        Monetary mortality or morbidity for each stove.
        """

        discount_rate, proj_life = self.discount_factor(model.specs)

        mor = {}
        cases_dict = {}
        costs_dict = {}

        diseases = ['alri', 'copd', 'ihd', 'lc', 'stroke']

        if relative:
            start_year = mask[mask].index
            self.health_parameters(model)

            cl_diseases = {'alri': {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06},
                           'copd': {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16},
                           'lc': {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23},
                           'ihd': {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23},
                           'stroke': {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}}

            for disease in diseases:
                rate = model.specs[f'{parameter}_{disease}']
                cl = 0
                i = model.year
                paf = f'paf_{disease.lower()}'
                pop, house, pop_increase = model.yearly_pop(model.year)
                mor[disease] = pd.Series(0, index=mask.index)
                mor[disease].loc[start_year] = pop.loc[start_year] * (model.base_fuel[paf].loc[start_year] - self[paf].loc[start_year]) * (
                        rate / 100000)
                cases = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                costs = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))

                while i <= model.specs["end_year"]:
                    if parameter == 'morb':
                        cost = model.specs[f'coi_{disease}']
                    elif parameter == 'mort':
                        cost = model.specs['vsl']

                    if i - model.year + 1 > 5:
                        cl = 1
                    else:
                        cl += cl_diseases[disease][i - model.year + 1]

                    disease_cases = mor[disease].loc[start_year] * cl * (1 + pop_increase.loc[start_year]) ** (
                            i - model.year)
                    cases[mask, i - model.specs["start_year"] - 1] += disease_cases.loc[start_year]
                    costs[mask, i - model.specs["start_year"] - 1] += disease_cases.loc[start_year] * cost

                    i += 1

                cases_dict[disease] = cases
                costs_dict[disease] = costs

            total_costs = np.sum(list(costs_dict.values()), axis=0)
            total_cases = np.sum(list(cases_dict.values()), axis=0)

            discounted_costs = np.array([sum(x / discount_rate) for x in total_costs])
        else:
            for disease in diseases:
                i = model.specs["start_year"]
                pop, house, pop_increase = model.yearly_pop(model.specs["start_year"])
                rate = model.specs[f'{parameter}_{disease}']
                paf = f'paf_{disease.lower()}'
                mor[disease] = (pop * (self[paf]) * (rate / 100000))/house
                cases = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                costs = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                while i < model.specs["end_year"]:
                    if parameter == 'morb':
                        cost = model.specs[f'coi_{disease}']
                    elif parameter == 'mort':
                        cost = model.specs['vsl']

                    disease_cases = mor[disease] * (1 + pop_increase) ** (i - model.specs["start_year"])
                    cases[:, i - model.specs["start_year"]] += disease_cases
                    costs[:, i - model.specs["start_year"]] += (disease_cases * cost)

                    i += 1

                cases_dict[disease] = cases
                costs_dict[disease] = costs

            total_costs = np.sum(list(costs_dict.values()), axis=0)
            total_cases = np.sum(list(cases_dict.values()), axis=0)
            discounted_costs = np.array([sum(x / discount_rate) for x in total_costs])

        return total_cases, total_costs, discounted_costs

    def mortality(self, model: 'onstove.OnStove', mask, parameter, relative):
        """
        Distributes the total mortality across the study area per fuel.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        mask: gpd.GeoSeries
            Determines which cells are ran. This is relevant when the start and end years are different

        See also
        --------
        mort_morb

        """

        deaths, costs, discounted_costs = self.mort_morb(model, mask, parameter=parameter, relative=relative)

        masky = mask[mask].index

        if relative:

            if not isinstance(self.deaths_avoided, pd.Series):
                self.relative_deaths = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                self.deaths_avoided = pd.Series(0, index=mask.index)
                self.distributed_mortality = pd.Series(0, index=mask.index)
            self.distributed_mortality.loc[masky] = pd.Series(discounted_costs, index=mask.index).loc[masky]
            self.relative_deaths[mask] = deaths[mask]
            self.deaths_avoided.loc[masky] = \
                pd.Series(np.array([sum(x) for x in self.relative_deaths]), index=mask.index).loc[masky]

            if model.specs['health_spillovers_parameter'] > 0:
                self.deaths_avoided.loc[masky] = self.deaths_avoided.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.distributed_mortality.loc[masky] = self.distributed_mortality.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
        else:
            if not isinstance(self.deaths, np.ndarray):
                self.deaths = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                self.distributed_mortality = pd.Series(0, index=mask.index)
                self.mort_costs = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))

            self.deaths[mask] = deaths[mask]
            self.mort_costs[mask] = costs[mask]
            self.distributed_mortality.loc[masky] = pd.Series(discounted_costs, index=mask.index).loc[masky]
            if model.specs['health_spillovers_parameter'] > 0:
                self.mort_costs[mask] = self.mort_costs[mask] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.deaths[mask] = self.deaths[mask] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.distributed_mortality.loc[masky] = self.distributed_mortality.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])

    def morbidity(self, model: 'onstove.OnStove', mask, parameter, relative):
        """
        Distributes the total morbidity across the study area per fuel.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        mask: gpd.GeoSeries
            Determines which cells are ran. This is relevant when the start and end years are different

        See also
        --------
        mort_morb
        """

        cases, costs, discounted_costs = self.mort_morb(model, mask, parameter=parameter, relative=relative)
        masky = mask[mask].index

        if relative:
            if not isinstance(self.cases_avoided, pd.Series):
                self.relative_cases = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                self.cases_avoided = pd.Series(0, index=mask.index)
                self.distributed_morbidity = pd.Series(0, index=mask.index)

            self.distributed_morbidity.loc[masky] = pd.Series(discounted_costs, index=mask.index).loc[masky]

            self.relative_cases[mask] = cases[mask]
            self.cases_avoided.loc[masky] = \
                pd.Series(np.array([sum(x) for x in self.relative_cases]), index=mask.index).loc[masky]

            if model.specs['health_spillovers_parameter'] > 0:
                self.cases_avoided.loc[masky] = self.cases_avoided.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.distributed_morbidity.loc[masky] = self.distributed_morbidity.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
        else:

            if not isinstance(self.cases, np.ndarray):
                self.cases = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))
                self.distributed_morbidity = pd.Series(0, index=mask.index)
                self.morb_costs = np.zeros((len(mask), model.specs["end_year"] - model.specs["start_year"]))

            self.distributed_morbidity.loc[masky] = pd.Series(discounted_costs, index=mask.index).loc[masky]

            self.cases[mask] = cases[mask]
            self.morb_costs[mask] = costs[mask]

            if model.specs['health_spillovers_parameter'] > 0:
                self.morb_costs[mask] = self.morb_costs[mask] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.cases[mask] = self.cases[mask] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])
                self.distributed_morbidity.loc[masky] = self.distributed_morbidity.loc[masky] * (1 + model.specs['w_spillover']
                                                                         * model.specs['health_spillovers_parameter'])

    def salvage(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """
        Calls discount_factor function and calculates discounted salvage cost for each stove assuming a straight-line depreciation.

                Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`

        See also
        --------
        discount_factor
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        inv = self.inv_cost * np.ones(mask.shape[0])

        if relative:
            year = model.year
        else:
            year = model.specs["start_year"]

        used_life = ((model.specs["end_year"] - year + 1) % self.tech_life) * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        proj_years[:, -1] = 1

        investments = proj_years * np.array(inv)[:, None]
        salvage = np.array([y * (1 - x / self.tech_life) for x, y in zip(used_life, investments)])

        start_year = mask[mask].index

        if relative:
            salvage_base = np.zeros(mask.shape[0])

            for life in model.base_fuel.tech_life.unique():
                remainder = (model.year - model.specs["start_year"]) % life
                idx = model.base_fuel.tech_life == life
                idx2 = idx * mask
                rows = idx2[idx2].index
                salvage_base[idx2] = model.base_fuel.inv_cost.loc[rows] * remainder / life
            # TODO: save this one
            salvage[mask, (model.year - model.specs["start_year"] - 1)] = salvage_base[mask]

            discounted_salvage = np.array([sum(x / discount_rate) for x in salvage])

            if not isinstance(self.discounted_salvage_cost, pd.Series):
                self.discounted_salvage_cost = pd.Series(0, index=mask.index, dtype='float64')

            self.discounted_salvage_cost.loc[start_year] = discounted_salvage[mask] \
                                                        - model.base_fuel.discounted_salvage_cost.loc[start_year]
        else:
            discounted_salvage = np.array([sum(x / discount_rate) for x in salvage])
            self.discounted_salvage_cost = pd.Series(discounted_salvage, index=mask.index)

    def discounted_om(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """
        Calls discount_factor function and calculates discounted operation and maintenance cost for each stove.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.


        See also
        --------
        discount_factor
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        operation_and_maintenance = self.om_cost * np.ones(mask.shape[0])


        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        if relative:
            discounted_base_om = model.base_fuel.discounted_om_costs

            base_annual_om = discounted_base_om / (
                    (1 - 1 / (1 + model.specs["discount_rate"]) ** proj_life) / model.specs["discount_rate"])
            base_annual_om = np.repeat(np.expand_dims(base_annual_om, axis=1),
                                       model.year - model.specs["start_year"] - 1, axis=1)
            proj_years[mask, (model.year - model.specs["start_year"] - 1):proj_years.shape[1]] = 1

            om = proj_years * np.array(operation_and_maintenance)[:, None]
            # TODO: save this one
            om[mask, 0:(model.year - model.specs["start_year"] - 1)] = base_annual_om[mask]

            discounted_om = np.array([sum(x / discount_rate) for x in om])

            if not isinstance(self.discounted_om_costs, pd.Series):
                self.discounted_om_costs = pd.Series(0, index=mask.index, dtype='float64')

            self.discounted_om_costs.loc[start_year] = discounted_om[mask] \
                                                               - discounted_base_om.loc[start_year]
        else:
            proj_years[:] = 1

            om = proj_years * np.array(operation_and_maintenance)[:, None]

            discounted_om = np.array([sum(x / discount_rate) for x in om])

            self.discounted_om_costs = pd.Series(discounted_om, index=mask.index)

    def discounted_inv(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool):
        """
        Calls discount_factor function and calculates discounted investment cost. Uses proj_life and tech_life to determine
        number of necessary re-investments

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        mask: pd.Series
            Mask rows included in the calculations. For the base fuel all rows are used, but for different years,
            ceratin rows are masked out.
        relative: bool, default True
            Boolean parameter to indicate if the discounted investments will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        discount_factor
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        inv = self.inv_cost * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        if relative:
            model.base_fuel.inv_cost = model.base_fuel.discounted_investments / (
                    (1 - 1 / (1 + model.specs["discount_rate"]) ** proj_life) / model.specs["discount_rate"])

            model.base_fuel.tech_life = round(model.base_fuel.tech_life).astype(int)

            proj_years[mask, model.year - model.specs["start_year"] - 1] = 1
            for j in range(self.tech_life, proj_life, self.tech_life):
                if j + model.year - model.specs["start_year"] - 1 < proj_life:
                    proj_years[mask, j + model.year - model.specs["start_year"] - 1] = 1

            investments = proj_years * np.array(inv)[:, None]

            proj_base_year = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                                       np.expand_dims(np.zeros(proj_life), axis=0))

            for life in model.base_fuel.tech_life.unique():
                if life < model.year - model.specs["start_year"]:
                    idx = model.base_fuel.tech_life == life
                    idx2 = idx * mask
                    rows = idx2[idx2].index
                    proj_base_year[rows, life - 1] = 1
                    for j in range(2 * life, proj_life, life):
                        if j < model.year - model.specs["start_year"] - 1:
                            proj_base_year[rows, j] = 1

            base_investments = proj_base_year * np.array(model.base_fuel.inv_cost)[:, None]
            # TODO: save this one
            investments = investments + base_investments

            investments_discounted = np.array([sum(x / discount_rate) for x in investments])

            if not isinstance(self.discounted_investments, pd.Series):
                self.discounted_investments = pd.Series(0, index=mask.index, dtype='float64')

            self.discounted_investments.loc[start_year] = investments_discounted[mask] \
                                                               - model.base_fuel.discounted_investments.loc[start_year]
        else:
            proj_years[:, 0] = 1
            for j in range(self.tech_life, proj_life, self.tech_life):
                if j < proj_life:
                    proj_years[:, j] = 1

            investments = proj_years * np.array(inv)[:, None]

            investments_discounted = np.array([sum(x / discount_rate) for x in investments])

            self.discounted_investments = pd.Series(investments_discounted, index=mask.index)

    def discount_fuel_cost(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """
        Calls discount_factor function and calculates discounted fuel costs.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        relative: bool, default True
            Boolean parameter to indicate if the discounted fuel cost will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        discount_factor
        """
        self.required_energy(model)
        discount_rate, proj_life = self.discount_factor(model.specs)

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        cost = np.ones(mask.shape[0])

        if isinstance(self.transport_cost, pd.Series):
            transport_cost = self.transport_cost.loc[start_year]
        else:
            transport_cost = self.transport_cost
        cost[mask] = (self.energy * self.fuel_cost / self.energy_content + transport_cost)

        if relative:
            discounted_base_fuel_cost = model.base_fuel.discounted_fuel_cost

            base_annual_fuel_cost = discounted_base_fuel_cost / (
                    (1 - 1 / (1 + model.specs["discount_rate"]) ** proj_life) / model.specs["discount_rate"])
            base_annual_fuel_cost = np.repeat(np.expand_dims(base_annual_fuel_cost, axis=1),
                                        model.year - model.specs["start_year"] - 1, axis=1)

            proj_years[mask, (model.year - model.specs["start_year"] - 1):proj_years.shape[1]] = 1

            fuel_cost = proj_years * np.array(cost)[:, None]
            # TODO: save this one
            fuel_cost[mask, 0:(model.year - model.specs["start_year"] - 1)] = base_annual_fuel_cost[mask]

            fuel_cost_discounted = np.array([sum(x / discount_rate) for x in fuel_cost])

            if not isinstance(self.discounted_fuel_cost, pd.Series):
                self.discounted_fuel_cost = pd.Series(0, index=mask.index, dtype='float64')

            self.discounted_fuel_cost.loc[start_year] = fuel_cost_discounted[mask] \
                                                               - discounted_base_fuel_cost.loc[start_year]
        else:
            proj_years[:] = 1

            fuel_cost = proj_years * np.array(cost)[:, None]

            fuel_cost_discounted = np.array([sum(x / discount_rate) for x in fuel_cost])

            self.discounted_fuel_cost = pd.Series(fuel_cost_discounted, index=mask.index)

    def total_time(self, model: 'onstove.OnStove', mask):
        """
        Calculates total time used per year by taking into account time of cooking and time of fuel collection (if relevant)

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        if not isinstance(self.total_time_yr, pd.Series):
            self.total_time_yr = pd.Series(0, index=mask.index, dtype='float64')

        masky = mask[mask].index

        if isinstance(self.time_of_collection, pd.Series):
            time_of_collection = self.time_of_collection.loc[masky]
        else:
            time_of_collection = self.time_of_collection

        self.total_time_yr.loc[masky] = pd.Series((self.time_of_cooking + time_of_collection) * 365,
                                                  index=mask.loc[masky].index)

    def time_saved(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """
        Calculates time saved per year by adopting a new stove.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        mask: pd.Series
            Determines which cells are ran. This is relevant when the start and end years are different
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        self.total_time(model, mask)

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        if relative:
            start_year = mask[mask].index

            base_annual_time = np.repeat(np.expand_dims(model.base_fuel.total_time_yr, axis=1),
                                         model.specs["end_year"] - model.specs["start_year"], axis=1)

            proj_years[mask, (model.year - model.specs["start_year"] - 1):proj_years.shape[1]] = 1

            time = proj_years[mask] * np.array(self.total_time_yr.loc[start_year])[:, None]
            time[:, 0:(model.year - model.specs["start_year"] - 1)] = base_annual_time[mask,
                                                                               0:model.year - model.specs[
                                                                                   "start_year"] - 1]

            if not isinstance(self.time_value, pd.Series):
                self.total_time_saved = pd.Series(0, index=mask.index, dtype='float64')
                self.time_value = pd.Series(0, index=mask.index, dtype='float64')

            # TODO: save this one
            decreased_time = base_annual_time[mask] - time

            self.total_time_saved.loc[start_year] = decreased_time.sum(axis=1)

            discounted_time = np.array([sum(x / discount_rate) for x in decreased_time]) * model.gdf.loc[start_year]["value_of_time"]

            self.time_value.loc[start_year] = discounted_time[mask]
        else:
            proj_years[:] = 1
            time = proj_years * np.array(self.total_time_yr)[:, None]
            discounted_time = np.array([sum(x / discount_rate) for x in time]) * model.gdf["value_of_time"]
            self.discounted_time_value = pd.Series(discounted_time, index=mask.index)


    def total_costs(self, model: 'onstove.OnStove', masky):
        """
        Calculates total costs (fuel, investment, operation and maintenance as well as salvage costs)

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        discount_fuel_cost, discounted_om, salvage, discounted_inv
        """

        if not isinstance(self.costs, pd.Series):
            self.costs = pd.Series(0, index=masky.index)

        self.costs.loc[masky] = (self.discounted_fuel_cost.loc[masky] + self.discounted_investments.loc[masky] +
                      self.discounted_om_costs.loc[masky] - self.discounted_salvage_cost.loc[masky])

    def net_benefit(self, model: 'onstove.OnStove', mask):

        """This method combines all costs and benefits as specified by the user using the weights parameters

         Parameters
         ----------
         model: OnStove model
             Instance of the OnStove model containing the main data of the study case. See
             :class:`onstove.OnStove`.

         See also
         --------
         total_costs, morbidity, mortality, time_saved, carbon_emissions
        """
        masky = mask[mask].index
        self.total_costs(model, masky)

        if not isinstance(self.benefits, pd.Series):
            self.benefits = pd.Series(0, index=mask.index, dtype='float64')
            self.net_benefits = pd.Series(0, index=mask.index, dtype='float64')
            self.factor = pd.Series(0, index=mask.index, dtype='float64')
            self.households = pd.Series(0, index=mask.index, dtype='float64')


        self.benefits.loc[masky] = model.specs["w_health"] * (self.distributed_morbidity.loc[masky] + self.distributed_mortality.loc[masky]) + \
                        model.specs["w_environment"] * self.decreased_carbon_costs.loc[masky] + model.specs["w_time"] * self.time_value.loc[masky]
        self.net_benefits.loc[masky] = self.benefits.loc[masky] - model.specs["w_costs"] * self.costs.loc[masky]

        if "costs_{}".format(self.name) not in model.gdf.columns:
            model.gdf["costs_{}".format(self.name)] = np.nan
            model.gdf["benefits_{}".format(self.name)] = np.nan
            model.gdf["net_benefit_{}".format(self.name)] = np.nan

        model.gdf.loc[masky, "costs_{}".format(self.name)] = self.costs.loc[masky]
        model.gdf.loc[masky, "benefits_{}".format(self.name)] = self.benefits.loc[masky]
        model.gdf.loc[masky, "net_benefit_{}".format(self.name)] = self.net_benefits.loc[masky]
        self.factor = pd.Series(np.ones(mask.shape[0]), index=mask.index)
        pop, house, pop_increase = model.yearly_pop(model.year)
        self.households.loc[masky] = house.loc[masky]


class LPG(Technology):
    """LPG technology class used to model LPG stoves.

    This class inherits the standard :class:`Technology` class and is used to model stoves using LPG as fuel.
    The LPG is assumed to be bought either by the closest vendor or in the closest urban settlement depedning on
    data availability. In the first case a point layer indicating vendors is assumed to be passed to the OnStove after
    which a least-cost path is determined using a friction map. In the other case it is assumed that the traveltime map
    is passed to OnStove directly.

    Parameters
    ----------
    name: str, optional.
       Name of the technology to model.
    carbon_intensity: float, optional
       The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
       gas-specific intensities will be used (e.g. ch4_intensity).
    co2_intensity: float, default 63
       The CO2 emissions in kg/GJ of burned fuel.
    ch4_intensity: float, default 0.003
       The CH4 emissions in kg/GJ of burned fuel.
    n2o_intensity: float, default 0.0001
       The N2O emissions in kg/GJ of burned fuel.
    co_intensity: float, default 0
       The CO emissions in kg/GJ of burned fuel.
    bc_intensity: float, default 0.0044
       The black carbon emissions in kg/GJ of burned fuel.
    oc_intensity: float, default 0.0091
       The organic carbon emissions in kg/GJ of burned fuel.
    energy_content: float, default 45.5
       Energy content of the fuel in MJ/kg.
    tech_life: int, default 7
       Stove life in year.
    inv_cost: float, default 44
       Investment cost of the stove in USD.
    fuel_cost: float, default 1.04
       Fuel cost in USD/kg.
    time_of_cooking: float, default 2
       Daily average time spent for cooking with this stove in hours.
    om_cost: float, default 3.7
       Operation and maintenance cost in USD/year.
    efficiency: float, default 0.5
       Efficiency of the stove.
    pm25: float, default 43
       Particulate Matter emissions (PM25) in mg/kg of fuel.
    travel_time: Pandas Series, optional
       Pandas Series describing the time needed (in hours) to reach either the closest LPG supply point or urban
       settlement  from each population point. It is either calculated using the LPG supply points, friction layer and
       population density layer or taken directly from a traveltime map.
    truck_capacity: float, default 2000
       Capacity of the truck carrying the fuel in kg.
    diesel_cost: float, 1.04
       Cost of diesel used in order to estimate the cost of transportation. Given in USD/liter
    diesel_per_hour: float, default 14
       Average diesel consumption of the truck carrying the fuel. Measured in liter/h
    lpg_path: str, optional
        Path to the lpg supply points (point vector file).
    friction_path: str, optional
       Path to the friction raster file describing the time needed (in minutes) to travel one meter within each
       cell using motorized transport.
    cylinder_cost: float, default 2.78
       Cost of LPG cylinder. This is relevant for first time buyers currenty not having access to an LPG cylinder.
       Given in USD/kg
    cylinder_life: float, 15
       Lifetime of LPG cylinder, measured in years.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[float] = None,
                 co2_intensity: float = 63,
                 ch4_intensity: float = 0.003,
                 n2o_intensity: float = 0.0001,
                 co_intensity: float = 0,
                 bc_intensity: float = 0.0044,
                 oc_intensity: float = 0.0091,
                 energy_content: float = 45.5,
                 tech_life: int = 7,  # in years
                 inv_cost: float = 44,  # in USD
                 fuel_cost: float = 0.73,
                 time_of_cooking: float = 2,
                 om_cost: float = 3.7,
                 efficiency: float = 0.5,  # ratio
                 pm25: float = 43,
                 travel_time: Optional[pd.Series] = None,
                 truck_capacity: float = 2000,
                 diesel_cost: float = 1.04,
                 diesel_per_hour: float = 14,
                 lpg_path: Optional[str] = None,
                 friction_path: Optional[str] = None,
                 cylinder_cost: float = 34.75,  # USD/cylinder,
                 cylinder_life: float = 15):
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
        self.cylinder_cost = cylinder_cost
        self.cylinder_life = cylinder_life
        self.discounted_infra_cost = None

    def add_travel_time(self, model: 'onstove.OnStove', lpg_path: Optional[str] = None,
                        friction_path: Optional[str] = None, align: bool = False):
        """This method calculates the travel time needed to transport LPG.

        The travel time is calculated as the time needed (in hours) to reach the closest LPG supplier from each
        population point. It uses a point layer for LPG suppliers, a friction layer and a population density layer.

        Parameters
        ----------
        lpg_path: str
            Path to the LPG supply points.
        friction_path: str
            Path to the friction raster file describing the time needed (in minutes) to travel one meter within each
            cell using motorized transport.
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        align: bool, default False
            Boolean parameter to indicate if the friction layer need to be align with the population
            data in the `model`.
        """

        if lpg_path is None:
            if self.lpg_path is not None:
                lpg_path = self.lpg_path
            else:
                raise ValueError('A path to a LPG point layer must be passed or stored in the `lpg_path` attribute.')

        lpg = VectorLayer(self.name, 'Suppliers', path=lpg_path)

        if friction_path is None:
            if self.friction_path is not None:
                friction_path = self.friction_path
            else:
                raise ValueError('A path to a friction raster layer must be passed or stored in the `friction_path`'
                                 ' attribute.')

        friction = RasterLayer(self.name, 'Friction', path=friction_path, resample='average')

        if align:
            os.makedirs(os.path.join(model.output_directory, self.name, 'Suppliers'), exist_ok=True)
            lpg.reproject(model.base_layer.meta['crs'], os.path.join(model.output_directory, self.name, 'Suppliers'))
            friction.align(model.base_layer.path, os.path.join(model.output_directory, self.name, 'Friction'))

        lpg.friction = friction
        lpg.travel_time(create_raster=True)
        travel_time = 2 * model.raster_to_dataframe(lpg.distance_raster,
                                                         fill_nodata_method='interpolate', method='read')
        self.travel_time = pd.Series(travel_time, index=mask.index)

    def transportation_cost(self, model: 'onstove.OnStove'):
        """The cost of transporting LPG.

        Transportation cost = (2 * diesel consumption per h * national diesel price * travel time)/transported LPG

        Total cost = (LPG cost + Transportation cost)/efficiency of LPG stoves

        For more information about cost formula see [1]_.

        The function uses the following attributes of model: ``diesel_per_hour``, ``diesel_cost``, ``travel_time``,
        ``truck_capacity``, ``efficiency`` and ``energy_content``.

        References
        ----------
        .. [1] Szabó, S., Bódis, K., Huld, T. & Moner-Girona, M. Energy solutions in rural Africa: mapping
           electrification costs of distributed solar and diesel generation versus grid extension.
           Environ. Res. Lett. 6, 034002 (2011).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """

        transport_cost = (self.diesel_per_hour * self.diesel_cost * self.travel_time) / self.truck_capacity
        kg_yr = (model.specs["meals_per_day"] * 365 * model.energy_per_meal) / (
                self.efficiency * self.energy_content)  # energy content in MJ/kg
        transport_cost = transport_cost * kg_yr
        transport_cost[transport_cost < 0] = np.nan
        self.transport_cost = transport_cost

    def discount_fuel_cost(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """This method expands :meth:`discount_fuel_cost` when LPG is the stove assessed in order to ensure that the
        transportation costs are included

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        relative: bool, default True
            Boolean parameter to indicate if the discounted investments will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        discount_fuel_cost
        """
        self.transportation_cost(model)
        super().discount_fuel_cost(model, mask, relative)

    def transport_emissions(self, model: 'onstove.OnStove', mask: int):
        """Calculates the emissions caused by the transportation of LPG. This is dependent on the diesel consumption of
        the truck. Diesel consumption is assumed to be 14 l/h (14 l/100km). Each truck is assumed to transport 2,000
        kg LPG

        Emissions intensities and diesel density are taken from [1]_.

        The function uses the following attributes of model: ``energy``, ``energy_content``, ``travel_time`` and
        ``truck_capacity``.

        References
        ----------
        .. [1] Ntziachristos, L. and Z. Samaras (2018), “1.A.3.b.i, 1.A.3.b.ii, 1.A.3.b.iii, 1.A.3.b.iv Passenger cars,
           light commercial trucks, heavy-duty vehicles including buses and motor cycles”, in EMEP/EEA air pollutant
           emission inventory guidebook 2016 – Update Jul. 2018

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        Returns
        -------
        The total transport emissions that can be associated with each household in kg of CO2-eq per year.
        """
        diesel_density = 840  # kg/m3
        bc_fraction = 0.55  # BC fraction of pm2.5
        oc_fraction = 0.70  # OC fraction of BC
        pm_diesel = 1.52  # g/kg_diesel
        diesel_ef = {'co2': 3.169, 'co': 7.40, 'n2o': 0.056,
                     'bc': bc_fraction * pm_diesel, 'oc': oc_fraction * bc_fraction * pm_diesel}  # g/kg_Diesel
        kg_yr = self.energy / self.energy_content  # LPG use (kg/yr). Energy required (MJ/yr)/LPG energy content (MJ/kg)
        masky = mask[mask].index
        diesel_consumption = self.travel_time.loc[masky] * (14 / 1000) * diesel_density  # kg of diesel per trip
        hh_emissions = sum([ef * model.gwp[pollutant] * diesel_consumption / self.truck_capacity * kg_yr for
                            pollutant, ef in diesel_ef.items()])  # in gCO2eq per yr
        return hh_emissions / 1000  # in kgCO2eq per yr

    def carb(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.carbon` when LPG is the stove assessed in order to ensure that the
         emissions caused by the transportation is included.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        transport_emissions
        """
        super().carb(model, mask)
        masky = mask[mask].index
        self.carbon.loc[masky] += self.transport_emissions(model, mask)

    def infrastructure_cost(self, model: 'onstove.OnStove', mask: pd.Series):
        """Calculates cost of cylinders for first-time LPG users. It is assumed that the cylinder contains 12.5 kg of
        LPG. The function calls ``infrastructure_salvage``.

        The function uses the ``cylinder_cost`` attribute of the model.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        infrastructure_salvage
        """

        discount_rate, proj_life = self.discount_factor(model.specs)
        cost = self.cylinder_cost * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        proj_years[mask, model.year - model.specs["start_year"] - 1] = 1
        for j in range(self.cylinder_life, proj_life, self.cylinder_life):
            if j + model.year - model.specs["start_year"] - 1 < proj_life:
                proj_years[mask, j + model.year - model.specs["start_year"] - 1] = 1

        costs = proj_years * np.array(cost)[:, None]

        if not isinstance(self.discounted_infra_cost, pd.Series):
            self.discounted_infra_cost = pd.Series(0, index=mask.index, dtype='float64')

        costs_discounted = np.array([sum(x / discount_rate) for x in costs])

        salvage = self.infrastructure_salvage(model, cost, mask)

        self.discounted_infra_cost.loc[start_year] = costs_discounted[mask] - salvage[mask]

    def infrastructure_salvage(self, model: 'onstove.OnStove', cost: float, mask: pd.Series):
        """Calculates the salvaged cylinder cost. The function calls ``discount_factor``.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        cost: float
            The cost of buying an LPG-cylinder.


        Returns
        -------
        The discounted salvage cost of an LPG-cylinder.

        See also
        --------
        discount_factor
        """
        discount_rate, proj_life = self.discount_factor(model.specs)

        used_life = ((model.specs["end_year"] - model.year + 1) % self.cylinder_life) * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        proj_years[mask, -1] = 1

        investments = proj_years * np.array(cost)[:, None]
        salvage = np.array([y * (1 - x / self.cylinder_life) for x, y in zip(used_life, investments)])

        discounted_salvage = np.array([sum(x / discount_rate) for x in salvage])

        return discounted_salvage

    def discounted_inv(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """This method expands :meth:`Technology.discounted_inv` by adding the cylinder cost for households currently
        not using LPG.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        relative: bool, default True
            Boolean parameter to indicate if the discounted investments will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        infrastructure_cost
        """
        super().discounted_inv(model, mask, relative=relative)
        if relative:
            start_year = mask[mask].index
            self.infrastructure_cost(model, mask)
            self.discounted_investments.loc[start_year] += (self.discounted_infra_cost.loc[start_year] * (1 - self.pop_sqkm.loc[start_year]))

class Biomass(Technology):
    """Biomass technology class used to model traditional and improved stoves.

    This class inherits the standard :class:`Technology` class and is used to model traditional and Improved Cook
    Stoves (ICS) using biomass as fuel. The biomass can be either collected or purchased, which is indicated with the
    attribute ``collected_fuel``. Depending on the biomass type (e.g. fuelwood or pellets), the parameters passed to
    the class such as efficiency, energy content, pm25 and emissions need to be representative of the fuel-stove.
    Moreover, the ICS can be modelled as natural draft or forced draft options by specifying it with the
    ``draft_type`` attribute. If forced draft is used, then the class will consider and extra capital cost for a
    standard 6 watt solar panel in order to run the fan in unelectrified areas.

    Attributes
    ----------
    forest: object of type RasterLayer, optional
        This is the forest cover raster dataset read from the ``forest_path`` parameter. See the
        :class:`onstove.RasterLayer` class for more information.
    friction: str, optional
        This is the forest cover raster dataset read from the ``friction_path``.
    trips_per_yr: float
        The trips that a person per household needs to do to the nearest forest point, in order to collect the amount
        of biomass required for cooking in one year.

    Parameters
    ----------
    name: str, optional.
        Name of the technology to model.
    carbon_intensity: float, optional
        The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
        gas-specific intensities will be used (e.g. ch4_intensity).
    co2_intensity: float, default 112
        The CO2 emissions in kg/GJ of burned fuel.
    ch4_intensity: float, default 0.864
        The CH4 emissions in kg/GJ of burned fuel.
    n2o_intensity: float, default 0.0039
        The N2O emissions in kg/GJ of burned fuel.
    co_intensity: float, default 0
        The CO emissions in kg/GJ of burned fuel.
    bc_intensity: float, default 0.1075
        The black carbon emissions in kg/GJ of burned fuel.
    oc_intensity: float, default 0.308
        The organic carbon emissions in kg/GJ of burned fuel.
    energy_content: float, default 16
        Energy content of the fuel in MJ/kg.
    tech_life: int, default 2
        Stove life in year.
    inv_cost: float, default 0
        Investment cost of the stove in USD.
    fuel_cost: float, default 0
        Fuel cost in USD/kg if any.
    time_of_cooking: float, default 2.9
        Daily average time spent for cooking with this stove in hours.
    om_cost: float, default 0
        Operation and maintenance cost in USD/year.
    efficiency: float, default 0.12
        Efficiency of the stove.
    pm25: float, default 844
        Particulate Matter emissions (PM25) in mg/kg of fuel.
    forest_path: str, optional
        Path to the forest cover raster file.
    friction_path: str, optional
        Path to the friction raster file describing the time needed (in minutes) to travel one meter within each
        cell.
    travel_time: Pandas Series, optional
        Pandas Series describing the time needed (in hours) to reach the closest forest cover point from each
        population point. It is calculated using the forest cover, friction layer and population density layer.

        .. seealso::
           :meth:`transportation_time<onstove.Biomass.transportation_time>` and
           :meth:`total_time<onstove.Biomass.total_time>`

    collection_capacity: float, default 25
        Average wood collection capacity per person in kg/trip.
    collected_fuel: bool, default True
        Boolean indicating if the fuel is collected or purchased. If True, then the ``travel_time`` will be
        calculated. If False, the ``fuel_cost`` will be used and a travel and collection time disregarded.
    time_of_collection: float, default 2
        Time spent collecting biomass on a single trip (excluding travel time) in hours.
    draft_type: str, default 'natural'
        Whether the ICS uses a natural draft or a forced draft.
    forest_condition: Callable object (function or lambda function) with a numpy array as input, optional
        Function or lambda function describing which forest canopy cover to consider when assessing the potential
        points for biomass collection.

        .. code-block:: python
           :caption: **Example**: lambda function for canopy cover equal or over 30%

           >>> forest_condition = lambda  x: x >= 0.3

    Examples
    --------
    An OnStove Biomass class can be created by providing the technology input data on an `csv` file and calling the
    :meth:`read_tech_data<onstove.read_tech_data>` method of the
    :class:`onstove.OnStove>` class, or by passing all technology information in the script.

    Creating the technologies from a `csv` configuration file (see *link to examples or mendeley* for a example of the
    configuration file):

    >>> from onstove import OnStove
    ... model = OnStove(output_directory='output_directory')
    ... mode.read_tech_data(path_to_config='path_to_csv_file', delimiter=',')
    ... model.techs
    {'Biomass': {'Biomass': <onstove.Biomass at 0x2478e85ee80>}}

    Creating a Biomass technology in the script:

    >>> from onstove import OnStove
    ... from onstove import Biomass
    ... model = OnStove(output_directory='output_directory')
    ... biomass = Biomass(name='Biomass')  # we define the name and leave all other parameters with the default values
    ... model.techs['Biomass'] = biomass
    ... model.techs
    {'Biomass': {'Biomass': <onstove.Biomass at 0x2478e85ee80>}}
    """

    forest: Optional[RasterLayer] = None
    friction: Optional[RasterLayer] = None
    trips_per_yr: float = 0.0

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[float] = None,
                 co2_intensity: float = 112,
                 ch4_intensity: float = 0.864,
                 n2o_intensity: float = 0.0039,
                 co_intensity: float = 0,
                 bc_intensity: float = 0.1075,
                 oc_intensity: float = 0.308,
                 energy_content: float = 16,
                 tech_life: int = 2,
                 inv_cost: float = 0,
                 solar_panel_life: float = 2,
                 fuel_cost: float = 0,
                 time_of_cooking: float = 2.9,
                 om_cost: float = 0,
                 efficiency: float = 0.12,
                 pm25: float = 844,
                 forest_path: Optional[str] = None,
                 friction_path: Optional[str] = None,
                 travel_time: Optional[pd.Series] = None,
                 discounted_solar_panel_cost = None,
                 collection_capacity: float = 25,
                 collected_fuel: bool = True,
                 time_of_collection: float = 2,
                 solar_panel_cost: float = 7.5,
                 draft_type: str = 'natural',
                 forest_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """Instantiates the class either with default or user defined values for each class attribute.
        """
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=False)

        self.forest_condition = forest_condition
        self.travel_time = travel_time
        self.forest_path = forest_path
        self.friction_path = friction_path
        self.collection_capacity = collection_capacity
        self.discounted_solar_panel_cost = discounted_solar_panel_cost
        self.draft_type = draft_type
        self.solar_panel_life = solar_panel_life
        self.solar_panel_cost = solar_panel_cost
        self.collected_fuel = collected_fuel
        self.time_of_collection = time_of_collection
        self.solar_panel_adjusted: bool = False  #: boolean check to avoid adding the solar panel cost twice

    def __setitem__(self, idx, value):
        self.__dict__[idx] = value

    def transportation_time(self, friction_path: str, forest_path: str, model: 'onstove.OnStove', mask: pd.Series, align: bool = False):
        """This method calculates the travel time needed to gather biomass.

        The travel time is calculated as the time needed (in hours) to reach the closest forest cover point from each
        population point. It uses a forest cover layer, a friction layer and population density layer.

        Parameters
        ----------
        friction_path: str
            Path to the friction raster file describing the time needed (in minutes) to travel one meter within each
            cell.
        forest_path: str
            Path to the forest cover raster file.
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        align: bool, default False
            Boolean parameter to indicate if the forest cover and friction layers need to be align with the population
            data in the `model`.
        """
        self.forest = RasterLayer(self.name, 'Forest', path=forest_path, resample='mode')
        self.friction = RasterLayer(self.name, 'Friction', path=friction_path, resample='average')

        self.forest.friction = self.friction
        rows, cols = self.forest.start_points(condition=self.forest_condition)
        self.forest.travel_time(rows=rows, cols=cols, create_raster=True)

        travel_time = 2 * model.raster_to_dataframe(self.forest.distance_raster,
                                                    fill_nodata_method='interpolate', method='read')
        travel_time[travel_time > 7] = 7  # cap to max travel time based on literature
        self.travel_time = pd.Series(travel_time, index=mask.index)

    def total_time(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.total_time` when biomass is collected.

        It calculates the time needed for collecting biomass, based on the ``collection_capacity`` the
        ``energy_content`` of the fuel, the ``energy`` required for cooking a standard meal and the travel time to
        the nearest forest area.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        masky = mask[mask].index
        if self.collected_fuel:
            if not isinstance(self.travel_time, pd.Series):
                self.transportation_time(self.friction_path, self.forest_path, model, mask)

            self.trips_per_yr = self.energy / (self.collection_capacity * self.energy_content)

            if not isinstance(self.total_time_yr, pd.Series):
                self.total_time_yr = pd.Series(0, index=mask.index)

            self.total_time_yr.loc[masky] = self.time_of_cooking * 365 + \
                                 (self.travel_time.loc[masky] + self.time_of_collection) * self.trips_per_yr
        else:
            self.time_of_collection = 0
            super().total_time(model, mask)

    def get_carbon_intensity(self, model: 'onstove.OnStove'):
        """This method expands :meth:`Technology.get_carbon_intensity`.

        It excludes the CO2 emissions from the share of firewood that is sustainably harvested (i.e. it does not affect
        other emissions such as CH4) by using the fraction of Non-Renewable Biomass (fNRB).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        Notes
        -----
        For more information about fNRB see [1]_.

        References
        ----------
        .. [1] R. Bailis, R. Drigo, A. Ghilardi, O. Masera, The carbon footprint of traditional woodfuels,
           Nature Clim Change. 5 (2015) 266–272. https://doi.org/10.1038/nclimate2491.
        """
        intensity = self['co2_intensity']
        self['co2_intensity'] *= model.specs['fnrb']
        super().get_carbon_intensity(model)
        self['co2_intensity'] = intensity

    def solar_panel_investment(self, model: 'onstove.OnStove', mask: pd.Series, relative = True):
        """This method adds the cost of a solar panel to unelectrified areas.

        The stove can be modelled a ICS with natural draft or forced draft. This is achieved by specifying the
        ``draft_type`` attribute of the class. If forced draft is used, then the class will consider and extra capital
        cost for a standard 6 watt solar panel in order to run the fan in unelectrified areas. The cost used for the
        panel is 1.25 USD per watt.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        inv = self.solar_panel_cost * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        if relative:
            proj_years[mask, model.year - model.specs["start_year"] - 1] = 1
            for j in range(self.solar_panel_life, proj_life, self.solar_panel_life):
                if j + model.year - model.specs["start_year"] - 1 < proj_life:
                    proj_years[mask, j + model.year - model.specs["start_year"] - 1] = 1

            investments = proj_years * np.array(inv)[:, None]

            investments_discounted = np.array([sum(x / discount_rate) for x in investments])

            if not isinstance(self.discounted_solar_panel_cost, pd.Series):
                self.discounted_solar_panel_cost = pd.Series(0, index=mask.index, dtype='float64')

            self.discounted_solar_panel_cost.loc[start_year] = investments_discounted[mask]
        else:
            proj_years[:, 0] = 1
            for j in range(self.solar_panel_life, proj_life, self.solar_panel_life):
                if j < proj_life:
                    proj_years[:, j] = 1

            investments = proj_years * np.array(inv)[:, None]

            investments_discounted = np.array([sum(x / discount_rate) for x in investments])

            self.discounted_solar_panel_cost = pd.Series(investments_discounted, index=mask.index)

    def discounted_inv(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """This method expands :meth:`Technology.discounted_inv` by adding the solar panel cost in unlectrified areas.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        relative: bool, default True
            Boolean parameter to indicate if the discounted investments will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        solar_panel_investment
        """
        masky = mask[mask].index
        super().discounted_inv(model, mask, relative=relative)
        if self.draft_type.lower().replace('_', ' ') in ['forced', 'forced draft']:
            self.solar_panel_investment(model, mask, relative)
            self.discounted_investments.loc[masky] += self.discounted_solar_panel_cost.loc[masky]


class Charcoal(Technology):
    """Charcoal technology class used to model traditional and improved stoves.

    This class inherits the standard :class:`Technology` class and is used to model traditional and Improved Cook
    Stoves (ICS) using charcoal as fuel.

    Parameters
    ----------
    name: str, optional.
        Name of the technology to model.
    carbon_intensity: float, optional
        The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
        gas-specific intensities will be used (e.g. ch4_intensity).
    co2_intensity: float, default 121
        The CO2 emissions in kg/GJ of burned fuel.
    ch4_intensity: float, default 0.576
        The CH4 emissions in kg/GJ of burned fuel.
    n2o_intensity: float, default 0.001
        The N2O emissions in kg/GJ of burned fuel.
    co_intensity: float, default 0
        The CO emissions in kg/GJ of burned fuel.
    bc_intensity: float, default 0.1075
        The black carbon emissions in kg/GJ of burned fuel.
    oc_intensity: float, default 0.308
        The organic carbon emissions in kg/GJ of burned fuel.
    energy_content: float, default 30
        Energy content of the fuel in MJ/kg.
    tech_life: int, default 2
        Stove life in year.
    inv_cost: float, default 4
        Investment cost of the stove in USD.
    fuel_cost: float, default 0.09
        Fuel cost in USD/kg if any.
    time_of_cooking: float, default 2.6
        Daily average time spent for cooking with this stove in hours.
    om_cost: float, default 3.7
        Operation and maintenance cost in USD/year.
    efficiency: float, default 0.2
        Efficiency of the stove.
    pm25: float, default 256
        Particulate Matter emissions (PM25) in mg/kg of fuel.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[float] = None,
                 co2_intensity: float = 121,
                 ch4_intensity: float = 0.576,
                 n2o_intensity: float = 0.001,
                 co_intensity: float = 0,
                 bc_intensity: float = 0.1075,
                 oc_intensity: float = 0.308,
                 energy_content: float = 30,
                 tech_life: float = 2,  # in years
                 inv_cost: int = 4,  # in USD
                 fuel_cost: float = 0.09,
                 time_of_cooking: float = 2.6,
                 om_cost: float = 3.7,
                 efficiency: float = 0.2,  # ratio
                 pm25: float = 256):
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=False)

    def get_carbon_intensity(self, model: 'onstove.OnStove'):
        """This method expands :meth:`Technology.get_carbon_intensity`.

        It excludes the CO2 emissions from the share of firewood that is sustainably harvested (i.e. it does not affect
        other emissions such as CH4) by using the fraction of Non-Renewable Biomass (fNRB).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        Notes
        -----
        For more information about fNRB see [1]_.

        References
        ----------
        .. [1] R. Bailis, R. Drigo, A. Ghilardi, O. Masera, The carbon footprint of traditional woodfuels,
           Nature Clim Change. 5 (2015) 266–272. https://doi.org/10.1038/nclimate2491.
        """
        intensity = self['co2_intensity']
        self['co2_intensity'] *= model.specs['fnrb']
        super().get_carbon_intensity(model)
        self['co2_intensity'] = intensity

    def production_emissions(self, model: 'onstove.OnStove'):
        """Calculates the emissions caused by the production of Charcoal. The function uses emission factors with regards
        to CO2, CO, CH4, BC and OC as well as the ``energy`` and ``energy_content`` attributes of te model.
        Emissions factors for the production of charcoal are taken from [1]_.


        References
        ----------
        .. [1] Akagi, S. K., Yokelson, R. J., Wiedinmyer, C., Alvarado, M. J., Reid, J. S., Karl, T., Crounse, J. D.,
            & Wennberg, P. O. (2010). Emission factors for open and domestic biomass burning for use in atmospheric
            models. Atmospheric Chemistry and Physics Discussions. 10: 27523–27602., 27523–27602.
            https://www.fs.usda.gov/treesearch/pubs/39297

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        Returns
        -------
        The total charcoal production emissions that can be associated with each household measured in
        kg of CO2-eq per year.
        """
        emission_factors = {'co2': 1626, 'co': 255, 'ch4': 39.6, 'bc': 0.02, 'oc': 0.74}  # g/kg_Charcoal
        # Charcoal produced (kg/yr). Energy required (MJ/yr)/Charcoal energy content (MJ/kg)
        kg_yr = self.energy / self.energy_content
        hh_emissions = sum([ef * model.gwp[pollutant] * kg_yr for pollutant, ef in
                            emission_factors.items()])  # gCO2eq/yr
        return hh_emissions / 1000  # kgCO2/yr

    def carb(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.carbon` when Charcoal is the fuel used (both traditional stoves and ICS)
         in order to ensure that the emissions caused by the production and transportation is included in the total emissions.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        carbon
        """
        super().carb(model, mask)
        masky = mask[mask].index
        self.carbon.loc[masky] += self.production_emissions(model)


class Electricity(Technology):
    """Electricity technology class used to model electrical stoves.

    This class inherits the standard :class:`Technology` class and is used to model electrical stoves.

    Attributes
    ----------
    carbon_intensities: dict
        Carbon intensities of the different power plants used in the electricity generation mix.
    grid_capacity_costs: dict
        Costs of adding capacity of the different power plants used in the generation mix.
    grid_techs_life: dict
        Technology life of the different power plants used in the generation mix.

    Parameters
    ----------
    name: str, optional
        Name of the technology to model.
    carbon_intensity: float, optional
        The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
        gas-specific intensities will be used (e.g. ch4_intensity).
    energy_content: float, default 3.6
        Energy content in MJ/kWh.
    tech_life: int, default 10
        Stove life in year.
    connection_cost: float, defualt 0
        Cost of strengthening a household connection to enable electrical cooking.
    grid_capacity_cost: float, optional
        Cost of added capacity in the grid (USD/kW)
    inv_cost: float, default 36.3
        Investment cost of the stove in USD.
    fuel_cost: float, default 0.1
        Fuel cost in USD/kg if any.
    time_of_cooking: float, default 1.8
        Daily average time spent for cooking with this stove in hours.
    om_cost: float, default 3.7
        Operation and maintenance cost in USD/year.
    efficiency: float, default 0.85
        Efficiency of the stove.
    pm25: float, default 32
        Particulate Matter emissions (PM25) in mg/kg of fuel.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[str] = None,
                 energy_content: float = 3.6,
                 tech_life: int = 10,  # in years
                 inv_cost: float = 36.3,  # in USD
                 connection_cost: float = 0,  # cost of additional infrastructure
                 grid_capacity_cost: float = None,
                 fuel_cost: float = 0.1,
                 time_of_cooking: float = 1.8,
                 om_cost: float = 3.7,  # percentage of investement cost
                 efficiency: float = 0.85,  # ratio
                 pm25: float = 32):
        super().__init__(name, carbon_intensity, None, None, None,
                         None, None, None, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=True)
        # Carbon intensity of fossil fuel plants in kg/GWh
        self.generation = {}
        self.capacities = {}
        self.grid_capacity_cost = grid_capacity_cost
        self.tiers_path = None
        self.discounted_capacity_cost = None
        self.capacity_cost = None
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
        self.grid_techs_life = {'oil': 40, 'natural_gas': 30,
                                'biofuels_and_waste': 25,
                                'nuclear': 50, 'hydro': 60, 'coal': 40, 'wind': 22,
                                'solar': 25, 'geothermal': 30}

    def __setitem__(self, idx, value):
        if 'generation' in idx:
            self.generation[idx.lower().replace('generation_', '')] = value
        elif 'grid_capacity_cost' in idx:
            self.grid_capacity_cost = value
        elif 'capacity' in idx:
            self.capacities[idx.lower().replace('capacity_', '')] = value
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

    def get_capacity_cost(self, model: 'onstove.OnStove', mask: pd.Series):
        """This method determines the cost of electricity for each added unit of capacity (kW). The added capacity is
        assumed to be the same shares as the current installed capacity (i.e. if a country uses 10% coal powered power
        plants and 90% natural gas, the added capacity will consist of 10% coal and 90% natural gas)

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        self.required_energy(model)
        if self.tiers_path is None:
            add_capacity = 1
        else:
            # TODO: This is never used, dont worry abt it for now
            model.raster_to_dataframe(self.tiers_path, name='Electricity_tiers', method='sample')
            self.tiers = model.gdf['Electricity_tiers'].copy()
            add_capacity = (self.tiers < 3)


        # TODO: update this with the matrix, so that we know where we have inv and salvage. (grid_salvage) and then the code below the if/else (capacity_cost)

        self.get_grid_capacity_cost()
        self.get_grid_life()

        discount_rate, proj_life = self.discount_factor(model.specs)
        cost = self.grid_capacity_cost * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        proj_years[mask, model.year - model.specs["start_year"] - 1] = 1
        self.grid_life = round(self.grid_life)

        for j in range(self.grid_life, proj_life, self.grid_life):
            if j + model.year - model.specs["start_year"] - 1 < proj_life:
                proj_years[mask, j + model.year - model.specs["start_year"] - 1] = 1

        costs = proj_years * np.array(cost)[:, None]

        if not isinstance(self.discounted_capacity_cost, pd.Series):
            self.discounted_capacity_cost = pd.Series(0, index=mask.index, dtype='float64')

        discounted_capacity_cost = np.array([sum(x / discount_rate) for x in costs])

        salvage = self.grid_salvage(model, mask, cost)

        self.discounted_capacity_cost.loc[start_year] = discounted_capacity_cost[mask] - salvage[
            start_year.tolist()]

        self.capacity = self.energy / (3.6 * self.time_of_cooking * 365)
        if not isinstance(self.capacity_cost, pd.Series):
            self.capacity_cost = pd.Series(0, index=mask.index, dtype='float64')

        self.capacity_cost.loc[start_year] = self.capacity * self.discounted_capacity_cost.loc[start_year]

    def get_carbon_intensity(self, model: 'onstove.OnStove'):
        """This function determines the carbon intensity of generated electricity based on the power plant mix in the
        area of interest.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        """
        grid_emissions = sum([gen * self.carbon_intensities[fuel] for fuel, gen in self.generation.items()])
        grid_generation = sum(self.generation.values())
        self.carbon_intensity = grid_emissions / grid_generation * 1000  # to convert from Mton/PJ to kg/GJ

    def get_grid_capacity_cost(self):
        """This function determines the grid capacity cost in the area of interest."""
        self.grid_capacity_cost = sum(
            [self.grid_capacity_costs[fuel] * (cap / sum(self.capacities.values())) for fuel, cap in
             self.capacities.items()])

    def get_grid_life(self):
        """This function determines the grid capacity cost in the area of interest."""
        self.grid_life = sum(
            [self.grid_techs_life[tech] * (cap / sum(self.capacities.values())) for tech, cap in
             self.capacities.items()])

    def grid_salvage(self, model: 'onstove.OnStove', mask: pd.Series, cost, single: bool = False):
        """This method determines the salvage cost of the grid connected power plants.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        single: bool, default True
            Boolean parameter to indicate if there is only one grid_capacity_cost or several.
        """
        discount_rate, proj_life = self.discount_factor(model.specs)

        used_life = ((model.specs["end_year"] - model.year + 1) % self.grid_life) * np.ones(mask.shape[0])

        proj_years = np.matmul(np.expand_dims(np.ones(mask.shape[0]), axis=1),
                               np.expand_dims(np.zeros(proj_life), axis=0))

        start_year = mask[mask].index

        proj_years[mask, -1] = 1

        if single:
            used_life = proj_life % self.grid_life
            discounted_salvage = self.grid_capacity_cost * (1 - used_life / self.grid_life)
        else:
            investments = proj_years * np.array(cost)[:, None]
            salvage = np.array([y * (1 - x / self.grid_life) for x, y in zip(used_life, investments)])

            discounted_salvage = np.array([sum(x / discount_rate) for x in salvage])

        return discounted_salvage

    def carb(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.carbon` when electricity is the fuel used


         Parameters
         ----------
         model: OnStove model
             Instance of the OnStove model containing the main data of the study case. See
             :class:`onstove.OnStove`.

         See also
         --------
         carbon
         """
        if self.carbon_intensity is None:
            self.get_carbon_intensity(model)
        super().carb(model, mask)

    def discounted_inv(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        """This method expands :meth:`Technology.discounted_inv` by adding connection and added capacity costs.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        relative: bool, default True
            Boolean parameter to indicate if the discounted investments will be calculated relative to the `base_fuel`
            or not.

        See also
        --------
        get_capacity_cost
        """
        super().discounted_inv(model, mask, relative=relative)
        masky = mask[mask].index
        if relative:
            self.get_capacity_cost(model, mask)
            if isinstance(self.connection_cost, pd.Series):
                self.discounted_investments.loc[masky] += (self.connection_cost.loc[masky] + self.discounted_capacity_cost.loc[masky] * (1 - self.pop_sqkm.loc[masky]))
            else:
                self.discounted_investments.loc[masky] += (
                        self.connection_cost + self.discounted_capacity_cost.loc[masky] * (1 - self.pop_sqkm.loc[masky]))

    def net_benefit(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.net_benefit` by taking into account electricity availability
        in the calculations.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        net_benefit
        """
        super().net_benefit(model, mask)
        model.gdf.loc[model.gdf['Current_elec'] == 0, "net_benefit_{}".format(self.name)] = np.nan
        #TODO: Everywere where we have model.gdf["pop_init_year"] or model.gdf["pop_end_year"] see if it needs to be
        #TODO: replaced with the population in the actual year.
        masky = mask[mask].index
        self.factor.loc[masky] = model.elec_pop.loc[masky]
        pop, house, pop_increase = model.yearly_pop(model.year)
        self.households.loc[masky] = model.elec_pop.loc[masky] * house.loc[masky]

class MiniGrids(Electricity):
    """Mini-grids technology class used to model electrical stoves powered by mini-grids.

    This class inherits and modifies the :class:`Electricity` class.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[str] = None,
                 energy_content: float = 3.6,
                 tech_life: int = 10,  # in years
                 inv_cost: float = 36.3,  # in USD
                 connection_cost: float = 0,  # cost of additional infrastructure
                 grid_capacity_cost: float = None,
                 fuel_cost: float = 0.1,
                 time_of_cooking: float = 1.8,
                 om_cost: float = 3.7,  # percentage of investement cost
                 efficiency: float = 0.85,  # ratio
                 pm25: float = 32):
        super().__init__(name=name, carbon_intensity=carbon_intensity, energy_content=energy_content,
                         tech_life=tech_life, inv_cost=inv_cost, connection_cost=connection_cost,
                         grid_capacity_cost=grid_capacity_cost, fuel_cost=fuel_cost,
                         time_of_cooking=time_of_cooking, om_cost=om_cost, efficiency=efficiency, pm25=pm25)

        self.coverage = None
        self.potential = None

    @property
    def coverage(self) -> RasterLayer:
        """:class:`VectorLayer` object containing a vector dataset showing the areas of coverage of the mini-grids.

        This layer must contain the following columns:
        * `capacity`: installed capacity of the mini-grids
        * `households`: amount of households served by the mini-grids
        * `geometry`: polygons showing areas of coverage

        .. seealso::
            :meth:`calculate_potential`
        """
        return self._coverage

    @coverage.setter
    def coverage(self, layer):
        self._coverage = vector_setter(layer)

    def calculate_potential(self):
        """Calculates the potential of each mini-grid for supporting eCooking in each area.
        """
        pass

    def discounted_inv(self, model: 'onstove.OnStove', mask: pd.Series, relative: bool = True):
        pass

    def net_benefit(self, mask):
        super().net_benefit(model, mask)
        model.gdf.loc[self.potential == 0, "net_benefit_{}".format(self.name)] = np.nan


class Biogas(Technology):
    """Biogas technology class used to model biogas fueled stoves. This class inherits the standard
    :class:`Technology` class and is used to model stove using biogas as fuel. Biogas stoves are assumed to not
    be available in urban settlements as the collection of manure is assumed to be limited. If the fuel is assumed
    to be purchased changes can be made to the function called ``available_biogas``. Biogas is also assumed to be
    restricted based on temperature (an average yearly temperature below 10 degrees Celsius is assumed to lead to
    heavy drops of efficiency [1]_). Biogas production is also assumed to be a very water intensive process [2]_, hence
    areas under water stress are assumed restricted as well.

    References
    ----------
    .. [1] Lohani, S. P., Dhungana, B., Horn, H. & Khatiwada, D. Small-scale biogas technology and clean cooking fuel:
        Assessing the potential and links with SDGs in low-income countries – A case study of Nepal.
        Sustainable Energy Technologies and Assessments 46, 101301 (2021).
    .. [2] Bansal, V., Tumwesige, V. & Smith, J. U. Water for small-scale biogas digesters in sub-Saharan Africa.
        GCB Bioenergy 9, 339–357 (2017).

    Parameters
    ----------
    name: str, optional.
        Name of the technology to model.
    carbon_intensity: float, optional
        The CO2 equivalent emissions in kg/GJ of burned fuel. If this attribute is used, then none of the
        gas-specific intensities will be used (e.g. ch4_intensity).
    co2_intensity: float, default 0
        The CO2 emissions in kg/GJ of burned fuel.
    ch4_intensity: float, default 0.029
        The CH4 emissions in kg/GJ of burned fuel.
    n2o_intensity: float, default 0.0006
        The N2O emissions in kg/GJ of burned fuel.
    co_intensity: float, default 0
        The CO emissions in kg/GJ of burned fuel.
    bc_intensity: float, default 0.0043
        The black carbon emissions in kg/GJ of burned fuel.
    oc_intensity: float, default 0.0091
        The organic carbon emissions in kg/GJ of burned fuel.
    energy_content: float, default 22.8
        Energy content of the fuel in MJ/kg.
    tech_life: int, default 20
        Stove life in year.
    inv_cost: float, default 550
        Investment cost of the stove in USD.
    fuel_cost: float, default 0
        Fuel cost in USD/kg if any.
    time_of_cooking: float, default 2
        Daily average time spent for cooking with this stove in hours.
    time_of_collection: float, default 3
        Time spent collecting biomass on a single trip (excluding travel time) in hours.
    om_cost: float, default 3.7
        Operation and maintenance cost in USD/year.
    efficiency: float, default 0.4
        Efficiency of the stove.
    pm25: float, default 43
        Particulate Matter emissions (PM25) in mg/kg of fuel.
    digestor_eff: float, default 0.4
        Efficiency of the digestor.
    friction_path: str, optional
        Path to the friction raster file describing the time needed (in minutes) to travel one meter within each
        cell.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 carbon_intensity: Optional[float] = None,
                 co2_intensity: float = 0,
                 ch4_intensity: float = 0.029,
                 n2o_intensity: float = 0.0006,
                 co_intensity: float = 0,
                 bc_intensity: float = 0.0043,
                 oc_intensity: float = 0.0091,
                 energy_content: float = 22.8,
                 tech_life: int = 20,  # in years
                 inv_cost: float = 550,  # in USD
                 fuel_cost: float = 0,
                 time_of_cooking: float = 2,
                 time_of_collection: float = 3,
                 om_cost: float = 3.7,
                 efficiency: float = 0.4,  # ratio
                 pm25: float = 43,
                 digestor_eff: float = 0.4,
                 friction_path: Optional[str] = None):
        super().__init__(name, carbon_intensity, co2_intensity, ch4_intensity,
                         n2o_intensity, co_intensity, bc_intensity, oc_intensity,
                         energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25, is_clean=True)

        self.digestor_eff = digestor_eff
        self.friction_path = friction_path
        self.time_of_collection = time_of_collection
        self.water = None
        self.temperature = None

    def read_friction(self, model: 'onstove.OnStove', friction_path: str):
        """Reads a friction layer in min per meter (walking time per meter) and returns a pandas series with the values
        for each populated grid cell in hours per meter

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        friction_path: str
            Path to where the friction layer is stored.

        Returns
        -------
        A pandas series with the values for each populated grid cell in hours per meter
        """

        friction = RasterLayer(self.name, 'Friction', path=friction_path, resample='average')
        data = model.raster_to_dataframe(friction, fill_nodata_method='interpolate', method='read')
        return data / 60

    def required_energy_hh(self, model: 'onstove.OnStove'):
        """Determines the required annual energy needed for cooking taking into account the stove efficiency.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        Returns
        -------
        Required annual energy needed for cooking
        """

        self.required_energy(model)
        return self.energy / self.digestor_eff

    def get_collection_time(self, model: 'onstove.OnStove'):
        """Caluclates the daily time of collection based on friction (hour/meter), the available biogas energy from
        each cell (MJ/yr/meter, 1000000 represents meters per km2) and the required energy per household (MJ/yr)

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        """
        self.available_biogas(model)
        required_energy_hh = self.required_energy_hh(model)

        friction = self.read_friction(model, self.friction_path)

        time_of_collection = required_energy_hh * friction / (model.gdf["biogas_energy"] / 1000000) / 365
        time_of_collection[time_of_collection == float('inf')] = np.nan
        mean_value = time_of_collection.mean()
        time_of_collection[time_of_collection.isna()] = mean_value
        self.time_of_collection = time_of_collection

    def available_biogas(self, model: 'onstove.OnStove'):
        """Calculates the biogas production potential in liters per day. It currently takes into account 6 categories
        of livestock (cattle, buffalo, sheep, goat, pig and poultry). The biogas potential for each category is determined
        following the methodology outlined by Lohani et al.[1]_ This function also applies a restriction to biogas
        production with regards to urban areas, areas with temperature lower than 10 degrees[1]_ celsius and areas under
        water stress[2]_.

        References
        ----------
        .. [1] Lohani, S. P., Dhungana, B., Horn, H. & Khatiwada, D. Small-scale biogas technology and clean cooking
            fuel: Assessing the potential and links with SDGs in low-income countries – A case study of Nepal.
            Sustainable Energy Technologies and Assessments 46, 101301 (2021).
        .. [2] Bansal, V., Tumwesige, V. & Smith, J. U. Water for small-scale biogas digesters in sub-Saharan Africa.
            GCB Bioenergy 9, 339–357 (2017).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        """

        from_cattle = model.gdf["Cattles"] * 12 * 0.15 * 0.8 * 305
        from_buffalo = model.gdf["Buffaloes"] * 14 * 0.2 * 0.75 * 305
        from_sheep = model.gdf["Sheeps"] * 0.7 * 0.25 * 0.8 * 452
        from_goat = model.gdf["Goats"] * 0.6 * 0.3 * 0.85 * 450
        from_pig = model.gdf["Pigs"] * 5 * 0.75 * 0.14 * 470
        from_poultry = model.gdf["Poultry"] * 0.12 * 0.25 * 0.75 * 450

        model.gdf["available_biogas"] = ((from_cattle + from_buffalo + from_goat + from_pig + from_poultry +
                                          from_sheep) * self.digestor_eff / 1000) * 365

        # Temperature restriction
        if (self.temperature is not None) and ('Temperature' not in model.gdf.columns):
            if isinstance(self.temperature, str):
                self.temperature = RasterLayer('Biogas', 'Temperature', self.temperature)

            model.raster_to_dataframe(self.temperature, name="Temperature", method='read',
                                      fill_nodata_method='interpolate')
            model.gdf.loc[model.gdf["Temperature"] < 10, "available_biogas"] = 0
            model.gdf.loc[(model.gdf["IsUrban"] > 20), "available_biogas"] = 0

        # Water availability restriction
        if (self.water is not None) and ('Water' not in model.gdf.columns):
            if isinstance(self.water, str):
                self.water = VectorLayer('Biogas', 'Water scarcity', self.water, bbox=model.mask_layer.data)
            model.raster_to_dataframe(self.water, name="Water",
                                      fill_nodata_method='interpolate', method='read')
            model.gdf.loc[model.gdf["Water"] == 0, "available_biogas"] = 0

        # Available biogas energy per year in MJ (energy content in MJ/m3)
        model.gdf["biogas_energy"] = model.gdf["available_biogas"] * self.energy_content

    def recalibrate_livestock(self, model: 'onstove.OnStove', buffaloes: str, cattles: str, poultry: str,
                              goats: str, pigs: str, sheeps: str):
        """Recalibrates the livestock maps and adds them to the main dataframe. It currently takes into account 6 categories
        of livestock (cattle, buffalo, sheep, goat, pig and poultry).

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.
        buffaloes: str
            Path to the buffalo dataset.
        cattles: str
            Path to the cattle dataset.
        poultry: str
            Path to the poultry dataset.
        goats: str
            Path to the goat dataset.
        pigs: str
            Path to the pig dataset.
        sheeps: str
            Path to the sheep dataset.
        """
        paths = {
            'Buffaloes': buffaloes,
            'Cattles': cattles,
            'Poultry': poultry,
            'Goats': goats,
            'Pigs': pigs,
            'Sheeps': sheeps}

        for name, path in paths.items():
            layer = RasterLayer('Livestock', name,
                                path=path)
            model.raster_to_dataframe(layer, name=name, method='read',
                                      fill_nodata_method='interpolate')

    def total_time(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.total_time` by adding the biogas collection time

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        total_time
        """
        if not isinstance(self.time_of_collection, pd.Series):
            self.get_collection_time(model)
        super().total_time(model, mask)

    def net_benefit(self, model: 'onstove.OnStove', mask):
        """This method expands :meth:`Technology.net_benefit` by taking into account biogas availability
        in the calculations.

        Parameters
        ----------
        model: OnStove model
            Instance of the OnStove model containing the main data of the study case. See
            :class:`onstove.OnStove`.

        See also
        --------
        net_benefit
        """
        masky = mask[mask].index
        super().net_benefit(model, mask)
        required_energy_hh = self.required_energy_hh(model)
        model.gdf.loc[(model.gdf['biogas_energy'] < required_energy_hh), "benefits_{}".format(self.name)] = np.nan
        model.gdf.loc[(model.gdf['biogas_energy'] < required_energy_hh), "net_benefit_{}".format(self.name)] = np.nan
        pop_init, house_init, pop_increase_init = model.yearly_pop(model.specs["start_year"])
        factor = model.gdf['biogas_energy'] / (required_energy_hh * house_init)
        factor[factor > 1] = 1
        self.factor.loc[masky] = factor.loc[masky]
        pop, house, pop_increase = model.yearly_pop(model.year)
        self.households.loc[masky] = house.loc[masky] * factor.loc[masky]
