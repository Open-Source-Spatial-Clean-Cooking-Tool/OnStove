import numpy as np
import pandas as pd
import os
import rasterio

from math import exp

from rasterio.fill import fillnodata

from .layer import VectorLayer, RasterLayer
from .raster import interpolate


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
        self.om_cost = om_cost
        self.time_of_cooking = time_of_cooking
        self.efficiency = efficiency
        self.pm25 = pm25
        self.time_of_collection = 0
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

        discount_factor = (1 + specs["Discount_rate_tech"]) ** year

        return discount_factor, proj_life

    def required_energy(self, model):
        self.energy = model.specs["Meals_per_day"] * 365 * model.energy_per_meal / self.efficiency
        # discount_rate, proj_life = self.discount_factor(specs_file)
        # energy_needed = self.energy * np.ones(proj_life)
        # self.discounted_energy = (energy_needed / discount_rate)

    def carb(self, model):
        self.required_energy(model)
        self.carbon = (self.energy * self.carbon_intensity) / 1000

    def carbon_emissions(self, model):
        self.carb(model)
        proj_life = model.specs['End_year'] - model.specs['Start_year']
        carbon = model.specs["Cost of carbon emissions"] * (model.base_fuel.carbon - self.carbon) / 1000 / (
                1 + model.specs["Discount_rate"]) ** (proj_life)

        self.decreased_carbon_emissions = model.base_fuel.carbon - self.carbon
        self.decreased_carbon_costs = carbon

    def health_parameters(self, model):
        rr_alri, rr_copd, rr_ihd, rr_lc, rr_stroke = self.relative_risk()
        self.paf_alri = self.paf(rr_alri, 1 - model.specs['clean_cooking_access'])
        self.paf_copd = self.paf(rr_copd, 1 - model.specs['clean_cooking_access'])
        self.paf_ihd = self.paf(rr_ihd, 1 - model.specs['clean_cooking_access'])
        self.paf_lc = self.paf(rr_lc, 1 - model.specs['clean_cooking_access'])
        self.paf_stroke = self.paf(rr_stroke, 1 - model.specs['clean_cooking_access'])

    def mortality(self, model):
        """
        Calculates mortality rate per fuel

        Returns
        ----------
        Monetary mortality for each stove in urban and rural settings
        """
        self.health_parameters(model)

        mort_alri = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_alri - self.paf_alri) * (model.specs["Mort_ALRI"] / 100000)
        mort_copd = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_copd - self.paf_copd) * (model.specs["Mort_COPD"] / 100000)
        mort_ihd = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_ihd - self.paf_ihd) * (model.specs["Mort_IHD"] / 100000)
        mort_lc = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_lc - self.paf_lc) * (model.specs["Mort_LC"] / 100000)
        mort_stroke = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_stroke - self.paf_stroke) * (model.specs["Mort_STROKE"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_stroke = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        mort_vector = []
        while i < 6:
            mortality_alri = cl_alri[i] * model.specs["VSL"] * mort_alri / (1 + model.specs["Discount_rate"]) ** (i - 1)
            mortality_copd = cl_copd[i] * model.specs["VSL"] * mort_copd / (
                    1 + model.specs["Discount_rate"]) ** (i - 1)
            mortality_lc = cl_lc[i] * model.specs["VSL"] * mort_lc / (
                    1 + model.specs["Discount_rate"]) ** (i - 1)
            mortality_ihd = cl_ihd[i] * model.specs["VSL"] * mort_ihd / (
                    1 + model.specs["Discount_rate"]) ** (i - 1)
            mortality_stroke = cl_stroke[i] * model.specs["VSL"] * mort_stroke / (1 + model.specs["Discount_rate"]) ** (i - 1)

            mort_total = (1 + model.specs["Health_spillovers_parameter"]) * (
                    mortality_alri + mortality_copd + mortality_lc + mortality_ihd + mortality_stroke)

            mort_vector.append(mort_total)

            i += 1

        mortality = np.sum(mort_vector)

        #  Distributed mortality per household
        self.distributed_mortality = model.gdf["Calibrated_pop"] / (
                    model.gdf["Calibrated_pop"].sum() * model.gdf['Households']) * mortality
        #  Total deaths avoided
        self.deaths_avoided = (mort_alri + mort_copd + mort_lc + mort_ihd + mort_stroke) * (
                    model.gdf["Calibrated_pop"] / (model.gdf["Calibrated_pop"].sum() * model.gdf['Households']))

    def morbidity(self, model):
        """
        Calculates morbidity rate per fuel

        Returns
        ----------
        Monetary morbidity for each stove in urban and rural settings
        """
        self.health_parameters(model)

        morb_alri = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_alri - self.paf_alri) * (model.specs["Morb_ALRI"] / 100000)
        morb_copd = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_copd - self.paf_copd) * (model.specs["Morb_COPD"] / 100000)
        morb_ihd = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_ihd - self.paf_ihd) * (model.specs["Morb_IHD"] / 100000)
        morb_lc = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_lc - self.paf_lc) * (model.specs["Morb_LC"] / 100000)
        morb_stroke = model.gdf["Calibrated_pop"].sum() * (model.base_fuel.paf_stroke - self.paf_stroke) * (model.specs["Morb_STROKE"] / 100000)

        cl_copd = {1: 0.3, 2: 0.2, 3: 0.17, 4: 0.17, 5: 0.16}
        cl_alri = {1: 0.7, 2: 0.1, 3: 0.07, 4: 0.07, 5: 0.06}
        cl_lc = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_ihd = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}
        cl_stroke = {1: 0.2, 2: 0.1, 3: 0.24, 4: 0.23, 5: 0.23}

        i = 1
        morb_vector = []
        while i < 6:
            morbidity_alri = cl_alri[i] * model.specs["COI_ALRI"] * morb_alri / (1 + model.specs["Discount_rate"]) ** (
                    i - 1)
            morbidity_copd = cl_copd[i] * model.specs["COI_COPD"] * morb_copd / (1 + model.specs["Discount_rate"]) ** (
                    i - 1)
            morbidity_lc = cl_lc[i] * model.specs["COI_LC"] * morb_lc / (1 + model.specs["Discount_rate"]) ** (i - 1)
            morbidity_ihd = cl_ihd[i] * model.specs["COI_IHD"] * morb_ihd / (1 + model.specs["Discount_rate"]) ** (i - 1)
            morbidity_stroke = cl_stroke[i] * model.specs["COI_STROKE"] * morb_stroke / (1 + model.specs["Discount_rate"]) ** (
                    i - 1)

            morb_total = (1 + model.specs["Health_spillovers_parameter"]) * (
                    morbidity_alri + morbidity_copd + morbidity_lc + morbidity_ihd + morbidity_stroke)

            morb_vector.append(morb_total)

            i += 1

        morbidity = np.sum(morb_vector)

        self.distributed_morbidity = model.gdf["Calibrated_pop"] / (
                    model.gdf["Calibrated_pop"].sum() * model.gdf['Households']) * morbidity
        self.cases_avoided = (morb_alri + morb_copd + morb_lc + morb_ihd + morb_stroke) * (
                    model.gdf["Calibrated_pop"] / (model.gdf["Calibrated_pop"].sum() * model.gdf['Households']))

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

        salvage[-1] = self.inv_cost * (1 - used_life / self.tech_life)

        discounted_salvage = salvage.sum() / discount_rate

        self.discounted_salvage_cost = discounted_salvage

    def discounted_om(self, model):
        """
        Calls discount_factor function and creates discounted OM costs.
        Returns
        ----------
        discountedOM costs for each stove during the project lifetime
        """
        discount_rate, proj_life = self.discount_factor(model.specs)
        operation_and_maintenance = self.om_cost * np.ones(proj_life) * self.inv_cost
        operation_and_maintenance[0] = 0

        i = self.tech_life
        while i < proj_life:
            operation_and_maintenance[i] = 0
            i = i + self.tech_life

        discounted_om_cost = operation_and_maintenance.sum() / discount_rate

        self.discounted_om_costs = discounted_om_cost

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

        discounted_investments = investments / discount_rate
        # TODO: the + self.inv_cost  is a workaround to account for the investment in year 0
        self.discounted_investments = discounted_investments.sum() + self.inv_cost

    def discount_fuel_cost(self, model):
        self.required_energy(model)
        discount_rate, proj_life = self.discount_factor(model.specs)

        cost = (self.energy * self.fuel_cost / self.energy_content + self.transport_cost) * np.ones(model.gdf.shape[0])

        fuel_cost = [np.ones(proj_life) * x for x in cost]

        fuel_cost_discounted = np.array([sum(x / discount_rate) for x in fuel_cost])

        self.discounted_fuel_cost = pd.Series(fuel_cost_discounted, index=model.gdf.index)

    def total_time(self, model):
        self.total_time_yr = (self.time_of_cooking + self.time_of_collection) * 365

    def time_saved(self, model):
        if self.is_base:
            self.total_time_saved = np.zeros(model.gdf.shape[0])
            self.time_value = np.zeros(model.gdf.shape[0])
        else:
            proj_life = model.specs['End_year'] - model.specs['Start_year']
            self.total_time(model)
            self.total_time_saved = model.base_fuel.total_time_yr - self.total_time_yr  # time saved per household
            # time value of time saved per sq km
            self.time_value = self.total_time_saved * model.gdf["value_of_time"] / (
                    1 + model.specs["Discount_rate"]) ** (proj_life)

    def total_costs(self):
        self.costs = (self.discounted_fuel_cost + self.discounted_investments + #- self.time_value +
                      self.discounted_om_costs - self.discounted_salvage_cost)

    def net_benefit(self, model, w_health=1, w_environment=1, w_social=1, w_costs=1):
        self.total_costs()
        self.benefits = w_health * (self.distributed_morbidity + self.distributed_mortality) + w_environment * self.decreased_carbon_costs + w_social * self.time_value
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
                 carbon_intensity=0,
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
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
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
        self.transport_cost = pd.Series(transport_cost[model.rows, model.cols], index=model.gdf.index)

    def discount_fuel_cost(self, model):
        self.transportation_cost(model)
        super().discount_fuel_cost(model)


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
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=844,
                 forest_path=None,
                 friction_path=None,
                 travel_time=None,
                 collection_capacity=25):
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
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

        self.travel_time = 2 * pd.Series(forest.distance_raster.layer[model.rows, model.cols],
                                         index=model.gdf.index)

    def total_time(self, model):
        self.transportation_time(self.friction_path, self.forest_path, model)
        trips_per_yr = self.energy / (self.collection_capacity * self.energy_content)
        self.total_time_yr = self.time_of_cooking * model.specs['Meals_per_day'] * 365 + (
                self.travel_time + self.time_of_collection) * trips_per_yr


class Electricity(Technology):
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
                 connection_cost=0,  # cost of additional infrastructure
                 grid_capacity_cost=0,
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=41):
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
        # Carbon intensity of fossil fuel plants in kg/GWh
        self.generation = {}
        self.tiers_path = None
        self.connection_cost = connection_cost
        self.carbon_intensities = {'coal': 0.090374363, 'natural_gas': 0.050300655,
                                   'crude_oil': 0.070650288, 'heavy_fuel_oil': 0.074687989,
                                   'oil': 0.072669139, 'diesel': 0.069332823,
                                   'still_gas': 0.060849859, 'flared_natural_gas': 0.051855075,
                                   'waste': 0.010736111, 'biofuels_and_waste': 0.010736111,
                                   'nuclear': 0, 'hydro': 0, 'wind': 0,
                                   'solar': 0, 'other': 0}

    def __setitem__(self, idx, value):
        if 'generation' in idx:
            self.generation[idx.lower().replace('generation_', '')] = value
        elif 'carbon_intensity' in idx:
            self.carbon_intensities[idx.lower().replace('carbon_intensity_', '')] = value
        elif 'grid_capacity_cost' in idx:
            self.grid_capacity_cost = value
        elif 'connection_cost' in idx:
            self.connection_cost = value
        else:
            super().__setitem__(idx, value)

    def get_capacity_cost(self, model):
        # TODO: this line assumes if no tiers data is added, that all population settlements will need added capacity
        self.required_energy(model)
        if self.tiers_path is None:
            add_capacity = 1
        else:
            model.raster_to_dataframe(self.tiers_path, name='tiers', method='sample')
            self.tiers = model.gdf['tiers'].copy()
            del model.gdf['tiers']
            add_capacity = (self.tiers < 3)
        self.capacity = self.energy * add_capacity / (3.6 * self.time_of_cooking * 365)
        self.capacity_cost = self.capacity * self.grid_capacity_cost

    def get_carbon_intensity(self):
        grid_emissions = sum([gen * self.carbon_intensities[fuel] for fuel, gen in self.generation.items()])
        grid_generation = sum(self.generation.values())
        self.carbon_intensity = grid_emissions / grid_generation * 1000  # to convert from Mton/PJ to kg/GJ

    def carb(self, model):
        self.get_carbon_intensity()
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
                 carbon_intensity=0,
                 energy_content=0,
                 tech_life=0,  # in years
                 inv_cost=0,  # in USD
                 fuel_cost=0,
                 time_of_cooking=0,
                 om_cost=0,  # percentage of investement cost
                 efficiency=0,  # ratio
                 pm25=173,
                 utilization_factor=0.5,
                 digestor_eff=0.4,
                 friction_path=None,):
        super().__init__(name, carbon_intensity, energy_content, tech_life,
                         inv_cost, fuel_cost, time_of_cooking,
                         om_cost, efficiency, pm25)
        # TODO: Check what's the difference between these two factors
        self.utilization_factor = utilization_factor
        self.digestor_eff = digestor_eff
        self.friction_path = friction_path

    def read_friction(self, model, friction_path):
        friction = RasterLayer(self.name, 'Friction', layer_path=friction_path, resample='average')
        data = friction.layer[model.rows, model.cols]
        return (self.time_of_collection * 60)/data  # (h * 60 min/h) / (min/m)


    def available_biogas(self, model):
        # Biogas production potential in liters per day
        from_cattle = model.gdf["Cattles"] * 12 * 0.15 * 0.8 * 305
        from_buffalo = model.gdf["Buffaloes"] * 14 * 0.2 * 0.75 * 305
        from_sheep = model.gdf["Sheeps"] * 0.7 * 0.25 * 0.8 * 452
        from_goat = model.gdf["Goats"] * 0.6 * 0.3 * 0.85 * 450
        from_pig = model.gdf["Pigs"] * 5 * 0.75 * 0.14 * 470
        from_poultry = model.gdf["Poultry"] * 0.12 * 0.25 * 0.75 * 450

        fraction = self.read_friction(model, self.friction_path) / (1000000 * 0.2)
        self.fraction = fraction

        model.gdf["available_biogas"] = ((from_cattle + from_buffalo + from_goat + from_pig + from_poultry + \
                                                  from_sheep) * self.digestor_eff/1000) * 365

        model.gdf["m3_biogas_hh"] = fraction * model.gdf["available_biogas"]

        del model.gdf["Cattles"]
        del model.gdf["Buffaloes"]
        del model.gdf["Sheeps"]
        del model.gdf["Goats"]
        del model.gdf["Pigs"]
        del model.gdf["Poultry"]

    def available_energy(self, model, temp, water=None):
        self.required_energy(model)
        model.raster_to_dataframe(temp.layer, name="Temperature", method='read',
                                  nodata=temp.meta['nodata'], fill_nodata='interpolate')
        if isinstance(water, VectorLayer):
            model.raster_to_dataframe(water.layer, name="Water", method='read')
            model.gdf.loc[model.gdf["Water"] == 0, "m3_biogas_hh"] = 0

        model.gdf.loc[model.gdf["Temperature"] < 10, "m3_biogas_hh"] = 0
        model.gdf.loc[(model.gdf["IsUrban"] > 20), "m3_biogas_hh"] = 0

        model.gdf["biogas_energy"] = model.gdf["available_biogas"] * self.energy_content
        model.gdf["biogas_energy_hh"] = model.gdf["m3_biogas_hh"] * self.energy_content
        model.gdf.loc[(model.gdf["biogas_energy_hh"] < self.energy), "biogas_energy_hh"] = 0
        model.gdf.loc[(model.gdf["biogas_energy_hh"] == 0), "biogas_energy"] = 0

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

    def net_benefit(self, model, w_health=1, w_environment=1, w_social=1, w_costs=1):
        super().net_benefit(model, w_health, w_environment, w_social, w_costs)
        model.gdf.loc[(model.gdf['biogas_energy_hh'] == 0) & \
                      (model.gdf["net_benefit_{}".format(self.name)] > 0), "net_benefit_{}".format(self.name)] = 0
        factor = model.gdf['biogas_energy'] / (self.energy * model.gdf['Households'])
        factor[factor > 1] = 1
        self.households = model.gdf['Households'] * factor
