import os
from csv import DictReader

import psycopg2
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Any

from onsstove.technology import Technology, LPG, Biomass, Electricity
from .raster import *
from .layer import VectorLayer, RasterLayer


class DataProcessor:
    """
    Class containing the methods to perform a Multi Criteria Analysis
    of clean cooking access potential. It calculates a Clean Cooking Demand
    Index, a Clean Cooking Supply Index and a Clean Cooking Potential Index
    """
    conn = None
    base_layer = None

    def __init__(self, project_crs=None, cell_size=None, output_directory='output'):
        """
        Initializes the class and sets an empty layers dictionaries.
        """
        self.layers = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        self.output_directory = output_directory
        self.mask_layer = None

    def get_layers(self, layers):
        if layers == 'all':
            _layers = self.layers
        else:
            _layers = {}
            for category, names in layers.items():
                _layers[category] = {}
                for name in names:
                    _layers[category][name] = self.layers[category][name]
        return _layers

    def set_postgres(self, db, POSTGRES_USER, POSTGRES_KEY):
        """
        Sets a conecion to a PostgreSQL database

        Parameters
        ----------
        arg1 :
        """
        self.conn = psycopg2.connect(database=db,
                                     user=POSTGRES_USER,
                                     password=POSTGRES_KEY)

    def add_layer(self, category, name, layer_path, layer_type, query=None,
                  postgres=False, base_layer=False, resample='nearest',
                  normalization=None, inverse=False, distance=None,
                  distance_limit=float('inf')):
        """
        Adds a new layer (type VectorLayer or RasterLayer) to the MCA class

        Parameters
        ----------
        arg1 :
        """
        output_path = os.path.join(self.output_directory, category, name)
        os.makedirs(output_path, exist_ok=True)

        if layer_type == 'vector':
            if postgres:
                layer = VectorLayer(category, name, layer_path, conn=self.conn,
                                    normalization=normalization,
                                    distance=distance,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)
            else:
                layer = VectorLayer(category, name, layer_path,
                                    normalization=normalization,
                                    distance=distance,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)

        elif layer_type == 'raster':
            layer = RasterLayer(category, name, layer_path,
                                normalization=normalization, inverse=inverse,
                                distance=distance, resample=resample)

            if base_layer:
                if not self.cell_size:
                    self.cell_size = (layer.meta['transform'][0],
                                      abs(layer.meta['transform'][1]))
                if not self.project_crs:
                    self.project_crs = layer.meta['crs']

                cell_size_diff = abs(self.cell_size[0] - layer.meta['transform'][0]) / \
                                 layer.meta['transform'][0]

                if (layer.meta['crs'] != self.project_crs) or (cell_size_diff > 0.01):
                    layer.reproject(self.project_crs,
                                    output_path=output_path,
                                    cell_width=self.cell_size[0],
                                    cell_height=self.cell_size[1])

                self.base_layer = layer

        if category in self.layers.keys():
            self.layers[category][name] = layer
        else:
            self.layers[category] = {name: layer}

    def add_mask_layer(self, name, layer_path, postgres=False):
        """
        Adds a vector layer to self.mask_layer, which will be used to mask all
        other layers into is boundaries
        """
        if postgres:
            sql = f'SELECT * FROM {layer_path}'
            self.mask_layer = gpd.read_postgis(sql, self.conn)
        else:
            self.mask_layer = gpd.read_file(layer_path)

        if self.mask_layer.crs != self.project_crs:
            self.mask_layer.to_crs(self.project_crs, inplace=True)

    def mask_layers(self, datasets='all'):
        """
        Uses the previously added mask layer in self.mask_layer to mask all
        other layers to its boundaries
        """
        if not isinstance(self.mask_layer, gpd.GeoDataFrame):
            raise Exception('The `mask_layer` attribute is empty, please first ' + \
                            'add a mask layer using the `.add_mask_layer` method.')
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.mask(self.mask_layer, output_path)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.mask(self.mask_layer, output_path)

    def align_layers(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                if isinstance(layer, VectorLayer):
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(self.base_layer.path, output_path)
                else:
                    if name != self.base_layer.name:
                        layer.align(self.base_layer.path, output_path)
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(self.base_layer.path, output_path)

    def reproject_layers(self, datasets='all'):
        """
        Goes through all layer and call their `.reproject` method with the
        `project_crs` as argument
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.reproject(self.project_crs, output_path)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.reproject(self.project_crs, output_path)

    def get_distance_rasters(self, datasets='all'):
        """
        Goes through all layer and call their `.distance_raster` method
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.get_distance_raster(self.base_layer.path,
                                          output_path, self.mask_layer)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.get_distance_raster(self.base_layer.path,
                                                       output_path, self.mask_layer)

    def normalize_rasters(self, datasets='all'):
        """
        Goes through all layer and call their `.normalize` method
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.distance_raster.normalize(output_path, self.mask_layer)

    @staticmethod
    def index(layers):
        weights = []
        rasters = []
        for name, layer in layers.items():
            rasters.append(layer.weight * layer.distance_raster.normalized.layer)
            weights.append(layer.weight)

        return sum(rasters) / sum(weights)

    def get_demand_index(self, datasets='all'):
        datasets = self.get_layers(datasets)['demand']
        layer = RasterLayer('Indexes', 'Demand_index', normalization='MinMax')
        layer.layer = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Demand Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer)
        layer.normalized.name = 'Demand Index'
        self.demand_index = layer.normalized

    def get_supply_index(self, datasets='all'):
        datasets = self.get_layers(datasets)['supply']
        layer = RasterLayer('Indexes', 'Supply index', normalization='MinMax')
        layer.layer = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Supply Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer)
        layer.normalized.name = 'Supply Index'
        self.supply_index = layer.normalized

    def get_clean_cooking_index(self, demand_weight=1, supply_weight=1):
        layer = RasterLayer('Indexes', 'Clean Cooking Potential Index', normalization='MinMax')
        layer.layer = (demand_weight * self.demand_index.layer + supply_weight * self.supply_index.layer) / \
                      (demand_weight + supply_weight)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Clean Cooking Potential Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer)
        layer.normalized.name = 'Clean Cooking Potential Index'
        self.clean_cooking_index = layer.normalized

    def save_datasets(self, datasets='all'):
        """
        Saves all layers that have not been previously saved
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.save(output_path)


class OnSSTOVE:
    """
    Class containing the methods to perform a geospatial max-benefit analysis
    on clean cooking access
    """
    conn = None
    base_layer = None

    population = 'path_to_file'

    def __init__(self, project_crs=None, cell_size=None, output_directory='output'):
        """
        Initializes the class and sets an empty layers dictionaries.
        """
        self.layers = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        self.output_directory = output_directory
        self.mask_layer = None
        self.specs = None
        self.techs = None
        self.base_fuel = None

    def get_layers(self, layers):
        if layers == 'all':
            layers = self.layers
        else:
            layers = {category: {name: self.layers[category][name]} for category, names in layers.items() for name in
                      names}
        return layers

    def set_postgres(self, db, POSTGRES_USER, POSTGRES_KEY):
        """
        Sets a conecion to a PostgreSQL database

        Parameters
        ----------
        arg1 : 
        """
        self.conn = psycopg2.connect(database=db,
                                     user=POSTGRES_USER,
                                     password=POSTGRES_KEY)

    def read_scenario_data(self, path_to_config: str, delimiter=','):
        """Reads the scenario data into a dictionary
        """
        config = {}
        with open(path_to_config, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=delimiter)
            config_file = list(reader)
            for row in config_file:
                if row['Value']:
                    if row['data_type'] == 'int':
                        config[row['Param']] = int(row['Value'])
                    elif row['data_type'] == 'float':
                        config[row['Param']] = float(row['Value'])
                    elif row['data_type'] == 'string':
                        config[row['Param']] = str(row['Value'])
                    else:
                        raise ValueError("Config file data type not recognised.")
        self.specs = config

    def read_tech_data(self, path_to_config: str, delimiter=','):
        """
        Reads the technology data from a csv file into a dictionary
        """
        techs = {}
        with open(path_to_config, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=delimiter)
            config_file = list(reader)
            for row in config_file:
                if row['Value']:
                    if row['Fuel'] not in techs:
                        if row['Fuel'] == 'LPG':
                            techs[row['Fuel']] = LPG()
                        elif 'biomass' in row['Fuel'].lower():
                            techs[row['Fuel']] = Biomass()
                        elif 'electricity' in row['Fuel'].lower():
                            techs[row['Fuel']] = Electricity()
                        else:
                            techs[row['Fuel']] = Technology()
                    if row['data_type'] == 'int':
                        techs[row['Fuel']][row['Param']] = int(row['Value'])
                    elif row['data_type'] == 'float':
                        techs[row['Fuel']][row['Param']] = float(row['Value'])
                    elif row['data_type'] == 'string':
                        techs[row['Fuel']][row['Param']] = str(row['Value'])
                    elif row['data_type'] == 'bool':
                        techs[row['Fuel']][row['Param']] = bool(row['Value'])
                    else:
                        raise ValueError("Config file data type not recognised.")
        for name, tech in techs.items():
            if tech.is_base:
                self.base_fuel = tech

        self.techs = techs

    def elec_current(self):
        elec_actual = self.specs['Elec_rate']
        urban_pop = (self.gdf.loc[self.gdf["IsUrban"] > 1, "Calibrated_pop"].sum())
        rural_pop = (self.gdf.loc[self.gdf["IsUrban"] <= 1, "Calibrated_pop"].sum())
        total_pop = self.gdf["Calibrated_pop"].sum()

        total_elec_ratio = self.specs["Elec_rate"]
        urban_elec_ratio = self.specs["urban_elec_rate"]
        rural_elec_ratio = self.specs["rural_elec_rate"]
        elec_modelled = 0

        factor = (total_pop * total_elec_ratio) / (urban_pop * urban_elec_ratio + rural_pop * rural_elec_ratio)
        urban_elec_ratio *= factor
        rural_elec_ratio *= factor

        if "Transformer_dist" in self.gdf.columns:
            self.gdf["Elec_dist"] = self.gdf["Transformer_dist"]
            priority = 1
            dist_limit = self.specs["Max_Transformer_dist"]
        elif "MV_line_dist" in self.gdf.columns:
            self.gdf["Elec_dist"] = self.gdf["MV_line_dist"]
            priority = 1
            dist_limit = self.specs["Max_MV_line_dist"]
        else:
            self.gdf["Elec_dist"] = self.gdf["HV_line_dist"]
            priority = 2
            dist_limit = self.specs["Max_HV_line_dist"]

        condition = 0

        min_night_lights = self.specs["Min_Night_Lights"]
        min_pop = self.specs["Min_Elec_Pop"]

        while condition == 0:
            urban_electrified = urban_pop * urban_elec_ratio
            rural_electrified = rural_pop * rural_elec_ratio

            if priority == 1:

                print(
                    'We have identified the existence of transformers or MV lines as input data; '
                    'therefore we proceed using those for the calibration')

                self.gdf["Elec_pop_calib"] = 0
                self.gdf["Current_elec"] = 0

                self.gdf.loc[
                    (self.gdf["Elec_dist"] < dist_limit) & (self.gdf["Night_lights"] > min_night_lights) & (
                            self.gdf["Calibrated_pop"] > min_pop), "Elec_pop_calib"] = self.gdf["Calibrated_pop"]
                self.gdf.loc[
                    (self.gdf["Elec_dist"] < dist_limit) & (self.gdf["Night_lights"] > min_night_lights) & (
                            self.gdf["Calibrated_pop"] > min_pop), "Current_elec"] = 1

                urban_elec_modelled = self.gdf.loc[
                    (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] > 1), "Elec_pop_calib"].sum()

                rural_elec_modelled = self.gdf.loc[
                    (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] <= 1), "Elec_pop_calib"].sum()

                urban_elec_factor = urban_elec_modelled / urban_electrified
                rural_elec_factor = rural_elec_modelled / rural_electrified

                if urban_elec_factor > 1:
                    self.gdf.loc[(self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] > 1), "Elec_pop_calib"] *= (
                            1 / urban_elec_factor)
                else:
                    i = 0
                    while urban_elec_factor <= 1:
                        if i < 10:
                            self.gdf.loc[
                                (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] == 2), "Elec_pop_calib"] *= 1.1
                            self.gdf["Elec_pop_calib"] = np.minimum(self.gdf["Elec_pop_calib"],
                                                                    self.gdf["Calibrated_pop"])
                            urban_elec_modelled = self.gdf.loc[
                                (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] == 2), "Elec_pop_calib"].sum()
                            urban_elec_factor = urban_elec_modelled / urban_electrified
                            i += 1
                        else:
                            break

                if rural_elec_factor > 1:
                    self.gdf.loc[(self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] <= 1), "Elec_pop_calib"] *= (
                            1 / rural_elec_factor)
                else:
                    i = 0
                    while rural_elec_factor <= 1:
                        if i < 10:
                            self.gdf.loc[
                                (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] < 2), "Elec_pop_calib"] *= 1.1
                            self.gdf["Elec_pop_calib"] = np.minimum(self.gdf["Elec_pop_calib"],
                                                                    self.gdf["Calibrated_pop"])
                            rural_elec_modelled = self.gdf.loc[
                                (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] < 2), "Elec_pop_calib"].sum()
                            rural_elec_factor = rural_elec_modelled / rural_electrified
                            i += 1
                        else:
                            break

                pop_elec = self.gdf.loc[self.gdf["Current_elec"] == 1, "Elec_pop_calib"].sum()
                elec_modelled = pop_elec / total_pop

                # REVIEW. Added new calibration step for pop not meeting original steps, if prev elec pop is too small
                i = 0
                td_dist_2 = 0.1
                while elec_actual - elec_modelled > 0.01:
                    pop_elec_2 = self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                              (self.gdf["Elec_dist"] < td_dist_2), "Calibrated_pop"].sum()
                    if i < 50:
                        if (pop_elec + pop_elec_2) / total_pop > elec_actual:
                            elec_modelled = (pop_elec + pop_elec_2) / total_pop
                            self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                         (self.gdf["Elec_dist"] < td_dist_2), "Elec_pop_calib"] = self.gdf[
                                "Calibrated_pop"]
                            self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                         (self.gdf["Elec_dist"] < td_dist_2), "Current_elec"] = 1
                        else:
                            i += 1
                            td_dist_2 += 0.1
                    else:
                        self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                     (self.gdf["Elec_dist"] < td_dist_2), "Elec_pop_calib"] = self.gdf[
                            "Calibrated_pop"]
                        self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                     (self.gdf["Elec_dist"] < td_dist_2), "Current_elec"] = 1
                        elec_modelled = (pop_elec + pop_elec_2) / total_pop
                        break

                if elec_modelled > elec_actual:
                    self.gdf["Elec_pop_calib"] *= elec_actual / elec_modelled
                pop_elec = self.gdf.loc[self.gdf["Current_elec"] == 1, "Elec_pop_calib"].sum()
                elec_modelled = pop_elec / total_pop

            # RUN_PARAM: Calibration parameters if only HV lines are available
            else:
                print(
                    'No transformers or MV lines were identified as input data; '
                    'therefore we proceed to the calibration with HV line info')
                self.gdf.loc[
                    (self.gdf["Elec_dist"] < dist_limit) & (self.gdf["Night_lights"] > min_night_lights) & (
                            self.gdf["Calibrated_pop"] > min_pop), "Current_elec"] = 1

                urban_elec_modelled = self.gdf.loc[
                    (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] > 1), "Elec_pop_calib"].sum()
                rural_elec_modelled = self.gdf.loc[
                    (self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] <= 1), "Elec_pop_calib"].sum()
                urban_elec_factor = urban_elec_modelled / urban_electrified
                rural_elec_factor = rural_elec_modelled / rural_electrified

                if urban_elec_factor > 1:
                    self.gdf.loc[(self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] > 1), "Elec_pop_calib"] *= (
                            1 / urban_elec_factor)
                else:
                    pass
                if rural_elec_factor > 1:
                    self.gdf.loc[(self.gdf["Current_elec"] == 1) & (self.gdf["IsUrban"] <= 1), "Elec_pop_calib"] *= (
                            1 / rural_elec_factor)
                else:
                    pass

                pop_elec = self.gdf.loc[self.gdf["Current_elec"] == 1, "Elec_pop_calib"].sum()
                elec_modelled = pop_elec / total_pop

                # REVIEW. Added new calibration step for pop not meeting original steps, if prev elec pop is too small
                i = 0
                td_dist_2 = 0.1
                while elec_actual - elec_modelled > 0.01:
                    pop_elec_2 = self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                              (self.gdf["Elec_dist"] < td_dist_2), "Calibrated_pop"].sum()
                    if i < 50:
                        if (pop_elec + pop_elec_2) / total_pop > elec_actual:
                            elec_modelled = (pop_elec + pop_elec_2) / total_pop
                            self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                         (self.gdf["Elec_dist"] < td_dist_2), "Elec_pop_calib"] = self.gdf[
                                "Calibrated_pop"]
                            self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                         (self.gdf["Elec_dist"] < td_dist_2), "Current_elec"] = 1
                        else:
                            i += 1
                            td_dist_2 += 0.1
                    else:
                        self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                     (self.gdf["Elec_dist"] < td_dist_2), "Elec_pop_calib"] = self.gdf[
                            "Calibrated_pop"]
                        self.gdf.loc[(self.gdf["Current_elec"] == 0) & (self.gdf["Calibrated_pop"] > min_pop) &
                                     (self.gdf["Elec_dist"] < td_dist_2), "Current_elec"] = 1
                        elec_modelled = (pop_elec + pop_elec_2) / total_pop
                        break

                if elec_modelled > elec_actual:
                    self.gdf["Elec_pop_calib"] *= elec_actual / elec_modelled
                pop_elec = self.gdf.loc[self.gdf["Current_elec"] == 1, "Elec_pop_calib"].sum()
                elec_modelled = pop_elec / total_pop

            urban_elec_ratio = self.gdf.loc[(self.gdf["Current_elec"] == 1) & (
                    self.gdf["IsUrban"] > 1), "Elec_pop_calib"].sum() / urban_pop
            rural_elec_ratio = self.gdf.loc[(self.gdf["Current_elec"] == 1) & (
                    self.gdf["IsUrban"] <= 1), "Elec_pop_calib"].sum() / rural_pop

            condition = 1

        self.gdf["Final_Elec_Code" + "{}".format(self.specs["Start_year"])] = \
            self.gdf.apply(lambda row: 1 if row["Current_elec"] == 1 else 99, axis=1)

    def calibrate_current_pop(self):

        total_gis_pop = self.gdf["Pop"].sum()
        calibration_factor = self.specs["Population_start_year"] / total_gis_pop

        self.gdf["Calibrated_pop"] = self.gdf["Pop"] * calibration_factor

    def distance_to_electricity(self, hv_lines=None, mv_lines=None, transformers=None):
        """
        Cals the get_distance_raster method for the HV. MV and Transformers
        layers (if available) and converts the output to a column in the
        main gdf
        """
        if (not hv_lines) and (not mv_lines) and (not transformers):
            raise ValueError("You MUST provide at least one of the following datasets: hv_lines, mv_lines or "
                             "transformers.")

        for layer in [hv_lines, mv_lines, transformers]:
            if layer:
                output_path = os.path.join(self.output_directory,
                                           layer.category,
                                           layer.name)
                layer.get_distance_raster(self.base_layer.path,
                                          output_path,
                                          self.mask_layer.layer)
                layer.distance_raster.layer /= 1000  # to convert from meters to km
                self.raster_to_dataframe(layer.distance_raster.layer,
                                         name=layer.distance_raster.name,
                                         method='read')

    def population_to_dataframe(self, layer):
        """
        Takes a population `RasterLayer` as input and extracts the populated points to a GeoDataFrame that is
        saved in `OnSSTOVE.gdf`.
        """
        self.rows, self.cols = np.where(~np.isnan(layer.layer))
        x, y = rasterio.transform.xy(layer.meta['transform'],
                                     self.rows, self.cols,
                                     offset='center')

        self.gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y),
                                     'Pop': layer.layer[self.rows, self.cols]})

    def raster_to_dataframe(self, layer, name=None, method='sample'):
        """
        Takes a RasterLayer and a method (sample or read), gets the values from the raster layer using the population points previously extracted and saves the values in a new column of OnSSTOVE.gdf
        """
        if method == 'sample':
            self.gdf[layer.name] = sample_raster(layer.path, self.gdf)
        elif method == 'read':
            self.gdf[name] = layer[self.rows, self.cols]

    def calibrate_urban_current_and_future_GHS(self):

        self.raster_to_dataframe(layer.urban.layer, name = "IsUrban")

        if self.specs["End_Year"] > self.specs["Start_Year"]:
            population_current = specs["Population_end_year"]
            urban_current = self.specs["Urban_start"] * population_current
            rural_current = population_current - urban_current

            population_future = specs["Population_end_year"]
            urban_future = self.specs["Urban_end"] * population_future
            rural_future = population_future - urban_future

            rural_growth = (rural_future - rural_current)/(self.specs["End_Year"] - self.specs["Start_Year"])
            urban_growth = (urban_future - urban_current) / (self.specs["End_Year"] - self.specs["Start_Year"])

            self.gdf.loc[self.gdf['IsUrban'] > 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * urban_growth
            self.gdf.loc[self.gdf['IsUrban'] < 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * rural_growth

    def calibrate_urban_manual(self):

        urban_modelled = 2
        factor = 1
        pop_tot = self.specs["Population_start_year"]
        urban_current = self.specs["Urban_start"]

        i = 0
        while abs(urban_modelled - urban_current) > 0.01:

            self.gdf["IsUrban"] = 0
            self.gdf.loc[(self.gdf["Calibrated_pop"] > 5000 * factor) & (
                    self.gdf["Calibrated_pop"] / (self.cell_size[0] ** 2 / 1000000) > 350 * factor), "IsUrban"] = 1
            self.gdf.loc[(self.gdf["Calibrated_pop"] > 50000 * factor) & (
                    self.gdf["Calibrated_pop"] / (self.cell_size[0] ** 2 / 1000000) > 1500 * factor), "IsUrban"] = 2

            pop_urb = self.gdf.loc[self.gdf["IsUrban"] > 1, "Calibrated_pop"].sum()

            urban_modelled = pop_urb / pop_tot

            if urban_modelled > urban_current:
                factor *= 1.1
            else:
                factor *= 0.9
            i = i + 1
            if i > 500:
                break


    def number_of_households(self):

        self.gdf.loc[self.gdf["IsUrban"] < 20, 'Households'] = self.gdf.loc[
                                                                    self.gdf["IsUrban"] < 20, 'Calibrated_pop'] / \
                                                                self.specs["Rural_HHsize"]
        self.gdf.loc[self.gdf["IsUrban"] > 20, 'Households'] = self.gdf.loc[self.gdf["IsUrban"] > 20, 'Calibrated_pop'] / \
                                                               self.specs["Urban_HHsize"]

    def get_value_of_time(self, wealth):
        """
        Calculates teh value of time based on the minimum wage ($/h) and a
        GIS raster map as wealth index, poverty or GDP
        ----
        0.5 is the upper limit for minimum wage and 0.2 the lower limit
        """
        wealth.layer[wealth.layer < 0] = np.nan
        wealth.meta['nodata'] = np.nan

        min_value = np.nanmin(wealth.layer)
        max_value = np.nanmax(wealth.layer)
        norm_layer = (wealth.layer - min_value) / (max_value - min_value) * (0.5 - 0.2) + 0.2
        self.value_of_time = norm_layer * self.specs['Minimum_wage'] / 30 / 24  # convert $/months to $/h
        self.raster_to_dataframe(self.value_of_time, name='value_of_time', method='read')

    def maximum_net_benefit(self):
        net_benefit_cols = [col for col in self.gdf if 'net_benefit_' in col]
        self.gdf["max_benefit_tech"] = self.gdf[net_benefit_cols].idxmax(axis=1)

        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("net_benefit_", "")
        self.gdf["maximum_net_benefit"] = self.gdf[net_benefit_cols].max(axis=1) #* self.gdf['Households']

        current_elect = self.gdf["Current_elec"] == 1
        elect_fraction = self.gdf.loc[current_elect, "Elec_pop_calib"] / self.gdf.loc[current_elect, "Calibrated_pop"]
        self.gdf.loc[current_elect, "maximum_net_benefit"] *= elect_fraction

        bool_vect = (self.gdf['Current_elec'] == 1) & (self.gdf['Elec_pop_calib'] < self.gdf['Calibrated_pop'])
        second_benefit_cols = [col for col in self.gdf if 'net_benefit_' in col]
        second_benefit_cols.remove('net_benefit_Electricity')
        second_best = self.gdf.loc[bool_vect, second_benefit_cols].idxmax(axis=1).str.replace("net_benefit_", "")

        current_biogas = (self.gdf["needed_energy"]*gdf["Households"] <= self.gdf["available_biogas_energy"]) &\
                         (self.gdf["max_benefit_tech"] == 'Biogas')
        biogas_fraction = self.gdf.loc[current_biogas, "available_biogas_energy"] / \
                          self.gdf.loc[current_biogas, "needed_energy"]
        self.gdf.loc[current_biogas, "maximum_net_benefit"] *= biogas_fraction

        biogas_bool = (self.gdf["needed_energy"]*gdf["Households"] <= self.gdf["available_biogas_energy"])
        second_benefit_cols = [col for col in self.gdf if 'net_benefit_' in col]
        second_benefit_cols.remove('net_benefit_Biogas')
        second_best_biogas = self.gdf.loc[biogas_bool, second_benefit_cols].idxmax(axis=1).str.replace("net_benefit_", "")

        second_best_value = self.gdf.loc[current_elect, second_benefit_cols].max(axis=1) * (1 - elect_fraction)
        second_tech_net_benefit = second_best_value #* self.gdf.loc[current_elect, 'Households']
        dff = self.gdf.loc[current_elect].copy()
        dff['max_benefit_tech'] = second_best
        dff['maximum_net_benefit'] = second_tech_net_benefit
        dff['Calibrated_pop'] *= (1 - elect_fraction)
        dff['Elec_pop_calib'] *= (1 - elect_fraction)
        dff['Households'] *= (1 - elect_fraction)

        self.gdf.loc[current_elect, 'Calibrated_pop'] *= elect_fraction
        self.gdf.loc[current_elect, 'Elec_pop_calib'] *= elect_fraction
        self.gdf.loc[current_elect, 'Households'] *= elect_fraction
        self.gdf = self.gdf.append(dff)

        benefit_cols = self.gdf.columns.str.contains('benefit')
        benefit_cols[np.where(self.gdf.columns == 'max_benefit_tech')] = False
        cost_cols = self.gdf.columns.str.contains('costs')
        columns = self.gdf.columns[benefit_cols | cost_cols]
        for col in columns:
            self.gdf[col] *= self.gdf['Households']

        second_best_value = self.gdf.loc[current_biogas, second_benefit_cols].max(axis=1) * (1 - biogas_fraction)
        second_tech_net_benefit = second_best_value #* self.gdf.loc[current_elect, 'Households']
        dff = self.gdf.loc[current_biogas].copy()
        dff['max_benefit_tech'] = second_best_biogas
        dff['maximum_net_benefit'] = second_tech_net_benefit
        dff['Calibrated_pop'] *= (1 - biogas_fraction)
        dff['Elec_pop_calib'] *= (1 - biogas_fraction)
        dff['Households'] *= (1 - biogas_fraction)

        self.gdf.loc[current_biogas, 'Calibrated_pop'] *= biogas_fraction
        self.gdf.loc[current_biogas, 'Biogas_pop_calib'] *= biogas_fraction
        self.gdf.loc[current_biogas, 'Households'] *= biogas_fraction
        self.gdf = self.gdf.append(dff)

        benefit_cols = self.gdf.columns.str.contains('benefit')
        benefit_cols[np.where(self.gdf.columns == 'max_benefit_tech')] = False
        cost_cols = self.gdf.columns.str.contains('costs')
        columns = self.gdf.columns[benefit_cols | cost_cols]
        for col in columns:
            self.gdf[col] *= self.gdf['Households']

    def lives_saved(self):
        self.gdf["deaths_avoided"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].deaths_avoided[row.name], axis=1) * self.gdf["Households"]

    def health_costs_saved(self):

        self.gdf["health_costs_avoided"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].distributed_morbidity[row.name] +
                        self.techs[row['max_benefit_tech']].distributed_mortality[row.name], axis=1) * self.gdf["Households"]

    def extract_time_saved(self):
        self.gdf["time_saved"] = self.gdf.apply(lambda row: self.techs[row['max_benefit_tech']].total_time_saved[row.name], axis=1) * \
                                 self.gdf["Households"]

    def reduced_emissions(self):

        self.gdf["reduced_emissions"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_emissions, axis=1) * self.gdf["Households"]

    def investment_costs(self):

        self.gdf["investment_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_investments, axis=1) * self.gdf["Households"]

    def om_costs(self):

        self.gdf["om_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_om_costs, axis=1) * self.gdf["Households"]

    def fuel_costs(self):

        self.gdf["fuel_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_fuel_cost[row.name], axis=1) * self.gdf["Households"]

    def emissions_costs_saved(self):

        self.gdf["emissions_costs_saved"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_costs, axis=1) * self.gdf["Households"]

    def gdf_to_csv(self, scenario_name):

        name = os.path.join(self.output_directory, scenario_name)

        pt = self.gdf.to_crs({'init': 'EPSG:3395'})

        pt["X"] = pt["geometry"].x
        pt["Y"] = pt["geometry"].y

        df = pd.DataFrame(pt.drop(columns='geometry'))
        df.to_csv(name)

    def to_raster(self, variable):
        layer = self.base_layer.layer.copy()
        tech_codes = None
        if isinstance(self.gdf[variable].iloc[0], str):
            dff = self.gdf.copy().reset_index(drop=False)
            dff[variable] += ' and '
            dff = dff.groupby('index').agg({variable: 'sum'})
            dff[variable] = [s[0:len(s) - 5] for s in dff[variable]]
            tech_codes = {tech: i for i, tech in enumerate(dff[variable].unique())}
            layer[self.rows, self.cols] = [tech_codes[tech] for tech in dff[variable]]
        else:
            dff = self.gdf.copy().reset_index(drop=False)
            dff = dff.groupby('index').agg({variable: 'sum'})
            layer[self.rows, self.cols] = dff[variable]

        raster = RasterLayer('Output', variable)
        raster.layer = layer
        raster.meta = self.base_layer.meta
        raster.save(os.path.join(self.output_directory, 'Output'))
        print(f'Layer saved in {os.path.join(self.output_directory, "Output", variable + ".tif")}\n')
        if tech_codes:
            print('Variable codes:')
            for tech, value in tech_codes.items():
                print('    ' + tech + ':', value)
            print('')
