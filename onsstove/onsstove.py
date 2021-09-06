import os
from csv import DictReader

import psycopg2
import geopandas as gpd
import numpy as np
from typing import Dict, Any

from onsstove.technology import Technology, LPG, Biomass
from .raster import *
from .layer import VectorLayer, RasterLayer


class OnSSTOVE():
    """
    Class containing the methods to perform a Multi Criteria Analysis
    of clean cooking access potential. It calculates a Clean Cooking Demand 
    Index, a Clean Cooking Supply Index and a Clean Cooking Potential Index
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

    def read_scenario_data(self, path_to_config: str):
        """Reads the scenario data into a dictionary
        """
        config = {}
        with open(path_to_config, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=';')
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

    def read_tech_data(self, path_to_config: str):
        """
        Reads the technology data from a csv file into a dictionary
        """
        techs = {}
        with open(path_to_config, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=';')
            config_file = list(reader)
            for row in config_file:
                if row['Value']:
                    if row['Fuel'] not in techs:
                        if row['Fuel'] == 'LPG':
                            techs[row['Fuel']] = LPG()
                        elif 'biomass' in row['Fuel'].lower():
                            techs[row['Fuel']] = Biomass()
                        else:
                            techs[row['Fuel']] = Technology()
                    if row['data_type'] == 'int':
                        techs[row['Fuel']][row['Param']] = int(row['Value'])
                    elif row['data_type'] == 'float':
                        techs[row['Fuel']][row['Param']] = float(row['Value'])
                    elif row['data_type'] == 'string':
                        techs[row['Fuel']][row['Param']] = str(row['Value'])
                    else:
                        raise ValueError("Config file data type not recognised.")
        self.techs = techs

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
                                    cell_height=self.cell_size[1],
                                    method=resample)

                self.base_layer = layer

        if category in self.layers.keys():
            self.layers[category][name] = layer
        else:
            self.layers[category] = {name: layer}

    def elec_current(self):

        urban_pop = (self.gdf.loc[self.gdf["IsUrban"] > 1, self.gdf["Calibrated_pop"]].sum())
        rural_pop = (self.gdf.loc[self.gdf["IsUrban"] <= 1, self.gdf["Calibrated_pop"]].sum())
        total_pop = self.gdf["Calibrated_pop"].sum()

        total_elec_ratio = self.specs["Elec_rate"]
        urban_elec_ratio = self.specs["urban_elec_rate"]
        rural_elec_ratio = self.specs["rural_elec_rate"]
        elec_modelled = 0

        factor = (total_pop * total_elec_ratio) / (urban_pop * urban_elec_ratio + rural_pop * rural_elec_ratio)
        urban_elec_ratio *= factor
        rural_elec_ratio *= factor

        self.gdf.loc[self.gdf["Night_lights"] == 0, self.gdf["Elec_pop_calib"]] = 0
        self.gdf["Current_elec"] = 0

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

    def calibrate_pop(self):

        total_gis_pop = self.gdf["Pop"].sum()
        calibration_factor = self.specs["Population_start_year"] / total_gis_pop

        self.gdf["Calibrated_pop"] = self.gdf["Pop"] * calibration_factor

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
                self.raster_to_dataframe(layer.distance_raster, method='read')

    def normalize_rasters(self, datasets='all'):
        """
        Goes through all layer and call their `.normalize` method
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.normalize(output_path, self.mask_layer)

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

    def calibrate_urban(self):

        urban_modelled = 2
        factor = 1
        pop_tot = self.specs["Population_start_year"]
        urban_current = self.specs["Urban_start"]

        i = 0
        while abs(urban_modelled - urban_current) > 0.01:

            self.gdf["IsUrban"] = 0
            self.gdf.loc[(self.gdf["Population"] > 5000 * factor) & (
                    self.gdf["Population"] / self.cell_size > 350 * factor), "IsUrban"] = 1
            self.gdf.loc[(self.gdf["Population"] > 50000 * factor) & (
                    self.gdf["Population"] / self.cell_size > 1500 * factor), "IsUrban"] = 2

            pop_urb = self.gdf.loc[clusters["IsUrban"] > 1, "Calibrated_pop"].sum()

            urban_modelled = pop_urb / pop_tot

            if urban_modelled > urban_current:
                factor *= 1.1
            else:
                factor *= 0.9
            i = i + 1
            if i > 500:
                break

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
        print(max_value, min_value)
        norm_layer = (wealth.layer - min_value) / (max_value - min_value) * (0.5 - 0.2) + 0.2
        self.value_of_time = norm_layer * self.specs['Minimum_wage']
        self.raster_to_dataframe(self.value_of_time, name='value_of_time', method='read')

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


    def maximum_net_benefit(self, df):

        net_benefit_cols = [col for col in df if 'final_tech' in col]
        df["final_tech"] = df[net_benefit_cols].idxmin(axis=1)
        df["maximum_net_benefit"] = df[net_benefit_cols].min(axis=1)

        df['final_tech'] = df['final_tech'].str.replace("net_benefit_","")