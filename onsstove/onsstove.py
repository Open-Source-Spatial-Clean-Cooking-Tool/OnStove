from copy import copy

import dill
from csv import DictReader

import psycopg2
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
from rasterio.warp import transform_bounds
from matplotlib.offsetbox import (TextArea, AnnotationBbox, VPacker, HPacker)

from plotnine import (
    ggplot,
    aes,
    geom_col,
    geom_text,
    ylim,
    scale_x_discrete,
    scale_fill_manual,
    scale_color_manual,
    coord_flip,
    theme_minimal,
    theme,
    labs,
    geom_boxplot,
    geom_density,
    after_stat,
    scale_y_log10
)

from onsstove.technology import Technology, LPG, Biomass, Electricity, Biogas, Charcoal
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
                  distance_limit=float('inf'), window=False, rescale=False):
        """
        Adds a new layer (type VectorLayer or RasterLayer) to the MCA class

        Parameters
        ----------
        arg1 :
        """

        if layer_type == 'vector':
            if postgres:
                layer = VectorLayer(category, name, layer_path, conn=self.conn,
                                    normalization=normalization,
                                    distance=distance,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)
            else:
                if window:
                    window = self.mask_layer.layer
                layer = VectorLayer(category, name, layer_path,
                                    normalization=normalization,
                                    distance=distance,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query, bbox=window)

        elif layer_type == 'raster':
            if window:
                with rasterio.open(layer_path) as src:
                    src_crs = src.meta['crs']
                if src_crs != self.mask_layer.layer.crs:
                    bounds = transform_bounds(self.mask_layer.layer.crs, src_crs, *self.mask_layer.bounds())
                window = bounds
            layer = RasterLayer(category, name, layer_path,
                                normalization=normalization, inverse=inverse,
                                distance=distance, resample=resample,
                                window=window, rescale=rescale)

            if base_layer:
                if not self.cell_size:
                    self.cell_size = (layer.meta['transform'][0],
                                      abs(layer.meta['transform'][4]))
                if not self.project_crs:
                    self.project_crs = layer.meta['crs']

                cell_size_diff = abs(self.cell_size[0] - layer.meta['transform'][0]) / \
                                 layer.meta['transform'][0]

                if (layer.meta['crs'] != self.project_crs) or (cell_size_diff > 0.01):
                    output_path = os.path.join(self.output_directory, category, name)
                    layer.reproject(self.project_crs,
                                    output_path=output_path,
                                    cell_width=self.cell_size[0],
                                    cell_height=self.cell_size[1])

                self.base_layer = layer

        if category in self.layers.keys():
            self.layers[category][name] = layer
        else:
            self.layers[category] = {name: layer}

    def add_mask_layer(self, category, name, layer_path, postgres=False, query=None):
        """
        Adds a vector layer to self.mask_layer, which will be used to mask all
        other layers into is boundaries
        """
        if postgres:
            self.mask_layer = VectorLayer(category, name, layer_path, self.conn, query)
        else:
            self.mask_layer = VectorLayer(category, name, layer_path, query=query)

        if self.mask_layer.layer.crs != self.project_crs:
            output_path = os.path.join(self.output_directory, category, name)
            self.mask_layer.reproject(self.project_crs, output_path)

    def mask_layers(self, datasets='all'):
        """
        Uses the previously added mask layer in self.mask_layer to mask all
        other layers to its boundaries
        """
        if not isinstance(self.mask_layer, VectorLayer):
            raise Exception('The `mask_layer` attribute is empty, please first ' + \
                            'add a mask layer using the `.add_mask_layer` method.')
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                if name != self.base_layer.name:
                    all_touched = True
                else:
                    all_touched = False
                layer.mask(self.mask_layer.layer, output_path, all_touched=all_touched)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.mask(self.mask_layer.layer, output_path)
                if isinstance(layer.distance_raster, RasterLayer):
                    layer.distance_raster.mask(self.mask_layer.layer, output_path)

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
                                          output_path, self.mask_layer.layer)
                # if isinstance(layer.friction, RasterLayer):
                #     layer.friction.get_distance_raster(self.base_layer.path,
                #                                        output_path, self.mask_layer.layer)

    def normalize_rasters(self, datasets='all', buffer=False):
        """
        Goes through all layer and call their `.normalize` method
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.distance_raster.normalize(output_path, self.mask_layer.layer, buffer=buffer)

    @staticmethod
    def index(layers):
        data = {}
        for k, i in layers.items():
            data.update(i)
        weights = []
        rasters = []
        for name, layer in data.items():
            rasters.append(layer.weight * layer.distance_raster.normalized.layer)
            weights.append(layer.weight)

        return sum(rasters) / sum(weights)

    def get_demand_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self.get_layers(datasets)
        layer = RasterLayer('Indexes', 'Demand_index', normalization='MinMax')
        layer.layer = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Demand Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.layer, buffer=buffer)
        layer.normalized.name = 'Demand Index'
        self.demand_index = layer.normalized

    def get_supply_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self.get_layers(datasets)
        layer = RasterLayer('Indexes', 'Supply index', normalization='MinMax')
        layer.layer = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Supply Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.layer, buffer=buffer)
        layer.normalized.name = 'Supply Index'
        self.supply_index = layer.normalized

    def get_clean_cooking_index(self, demand_weight=1, supply_weight=1, buffer=False):
        layer = RasterLayer('Indexes', 'Clean Cooking Potential Index', normalization='MinMax')
        layer.layer = (demand_weight * self.demand_index.layer + supply_weight * self.supply_index.layer) / \
                      (demand_weight + supply_weight)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Clean Cooking Potential Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.layer, buffer=buffer)
        layer.normalized.name = 'Clean Cooking Potential Index'
        self.clean_cooking_index = layer.normalized

    def get_assistance_need_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self.get_layers(datasets)
        layer = RasterLayer('Indexes', 'Assistance need index', normalization='MinMax')
        layer.layer = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Assistance need index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.layer, buffer=buffer)
        layer.normalized.name = 'Assistance need index'
        self.assistance_need_index = layer.normalized

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

    @staticmethod
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d}'.format(v=val) if (val / total) > 0.01 else ''

        return my_format

    def plot_share(self, index='clean cooking potential index', layer=('demand', 'population'),
                   title='Clean Cooking Potential Index', output_file=None):
        levels = []
        if index.lower() in 'clean cooking potential index':
            data = self.clean_cooking_index.layer
        elif index.lower() in 'assistance need index':
            data = self.assistance_need_index.layer
        elif index.lower() in 'supply index':
            data = self.supply_index.layer

        for level in [0.2, 0.4, 0.6, 0.8, 1]:
            levels.append(np.where(
                (data >= (level - 0.2)) & (data < level)))

        share = []
        for level in levels:
            value = np.nansum(self.layers[layer[0]][layer[1]].layer[level])
            if np.isnan(value):
                value = 0
            share.append(value)
        share.reverse()

        cmap = cm.get_cmap('magma_r', 5)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.pie(share,
               autopct=self.autopct_format(np.array(share) / 1000),
               pctdistance=1.2, textprops={'fontsize': 16},
               startangle=140,
               colors=cmap.colors)
        ax.legend(title='',
                  title_fontsize=16,
                  labels=['High', '', 'Medium', '', 'Low'],
                  bbox_to_anchor=(1.05, 0.8), borderaxespad=0.,
                  prop={'size': 16})
        ax.set_title(f'{title}\n(thousands)', loc='left', fontsize=18)

        if output_file:
            plt.savefig(os.path.join(self.output_directory, output_file),
                        dpi=150, bbox_inches='tight')

    def to_pickle(self, name):
        self.conn = None
        with open(os.path.join(self.output_directory, name), "wb") as f:
            dill.dump(self, f)

    @classmethod
    def read_model(cls, path):
        with open(path, "rb") as f:
            model = dill.load(f)
        return model


class OnSSTOVE(DataProcessor):
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
        super().__init__(project_crs, cell_size, output_directory)
        self.rows = None
        self.cols = None
        self.specs = None
        self.techs = None
        self.base_fuel = None
        self.i = {}
        self.energy_per_meal = 3.64
        # TODO: remove from here and make it an input in the specs file
        self.gwp = {'co2': 1, 'ch4': 25, 'n2o': 298, 'co': 2, 'bc': 900, 'oc': -46}

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
                    elif row['data_type'] == 'bool':
                        config[row['Param']] = str(row['Value']).lower() in ['true', 't', 'yes', 'y', '1']
                    else:
                        raise ValueError("Config file data type not recognised.")
        if self.specs is None:
            self.specs = config
        else:
            self.specs.update(config)

    def set_base_fuel(self, techs: list = None):
        """
        Defines the base fuel properties according to the technologies
        tagged as is_base = True or a list of technologies as input.
        If no technologies are passed as input and no technologies are tagged
        as is_base = True, then it calculates the base fuel properties considering
        all technologies in the model
        """
        if techs is None:
            techs = self.techs.values()
        base_fuels = []
        for tech in techs:
            if tech.is_base:
                base_fuels.append(tech)
        if len(base_fuels) == 1:
            self.base_fuel = copy(base_fuels[0])
            self.base_fuel.carb(self)
            self.base_fuel.total_time(self)
            self.base_fuel.required_energy(self)
            self.base_fuel.health_parameters(self)
            if isinstance(tech, LPG):
                self.base_fuel.transportation_cost(self)
            self.base_fuel.inv_cost = pd.Series([self.base_fuel.inv_cost] * self.gdf.shape[0],
                                                index=self.gdf.index)
            self.base_fuel.om_cost = pd.Series([self.base_fuel.om_cost] * self.gdf.shape[0],
                                                index=self.gdf.index)
        else:
            if len(base_fuels) == 0:
                base_fuels = techs
            base_fuel = Technology(name='Base fuel')
            base_fuel.carbon = 0
            base_fuel.total_time_yr = 0

            for tech in base_fuels:

                current_share = (self.gdf['IsUrban'] > 20) * tech.current_share_urban
                current_share[self.gdf['IsUrban'] < 20] = tech.current_share_rural


                tech.carb(self)
                tech.total_time(self)
                tech.required_energy(self)

                if isinstance(tech, LPG):
                    tech.transportation_cost(self)

                base_fuel.tech_life += tech.tech_life * current_share

                tech.health_parameters(self)

                base_fuel.carbon += tech.carbon * current_share
                base_fuel.total_time_yr += tech.total_time_yr * current_share
                base_fuel.inv_cost += tech.inv_cost * current_share
                base_fuel.om_cost += tech.om_cost * current_share

                tech.discount_fuel_cost(self, relative=False)
                base_fuel.discounted_fuel_cost += tech.discounted_fuel_cost * current_share

                for paf in ['paf_alri_r', 'paf_copd_r', 'paf_ihd_r',
                            'paf_lc_r', 'paf_stroke_r']:
                    base_fuel[paf] += tech[paf] * tech.current_share_rural
                for paf in ['paf_alri_u', 'paf_copd_u', 'paf_ihd_u',
                            'paf_lc_u', 'paf_stroke_u']:
                    base_fuel[paf] += tech[paf] * tech.current_share_urban
            self.base_fuel = base_fuel

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
                        if 'lpg' in row['Fuel'].lower():
                            techs[row['Fuel']] = LPG()
                        elif 'biomass' in row['Fuel'].lower():
                            techs[row['Fuel']] = Biomass()
                        elif 'charcoal' in row['Fuel'].lower():
                            techs[row['Fuel']] = Charcoal()
                        elif 'biogas' in row['Fuel'].lower():
                            techs[row['Fuel']] = Biogas()
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
                        techs[row['Fuel']][row['Param']] = str(row['Value']).lower() in ['true', 't', 'yes', 'y', '1']
                    else:
                        raise ValueError("Config file data type not recognised.")

        self.techs = techs

    def get_clean_cooking_access(self):
        clean_cooking_access_u = 0
        clean_cooking_access_r = 0
        for tech in self.techs.values():
            if tech.is_clean:
                clean_cooking_access_u += tech.current_share_urban
                clean_cooking_access_r += tech.current_share_rural
        self.clean_cooking_access_u = clean_cooking_access_u
        self.clean_cooking_access_r = clean_cooking_access_r

    def normalize(self, column, inverse=False):

        if inverse:
            normalized = (self.gdf[column].max() - self.gdf[column]) / (self.gdf[column].max() - self.gdf[column].min())
        else:
            normalized = (self.gdf[column] - self.gdf[column].min()) / (self.gdf[column].max() - self.gdf[column].min())

        return normalized

    def normalize_for_electricity(self):

        if "Transformers_dist" in self.gdf.columns:
            self.gdf["Elec_dist"] = self.gdf["Transformers_dist"]
        elif "MV_lines_dist" in self.gdf.columns:
            self.gdf["Elec_dist"] = self.gdf["MV_lines_dist"]
        else:
            self.gdf["Elec_dist"] = self.gdf["HV_lines_dist"]

        elec_dist = self.normalize("Elec_dist", inverse=True)
        ntl = self.normalize("Night_lights")
        pop = self.normalize("Calibrated_pop")

        self.combined_weight = (elec_dist * self.specs["infra_weight"] + pop * self.specs["pop_weight"] +
                                ntl * self.specs["NTL_weight"]) / (
                                       self.specs["infra_weight"] + self.specs["pop_weight"] +
                                       self.specs["NTL_weight"])

    def current_elec(self):
        self.normalize_for_electricity()
        elec_rate = self.specs["Elec_rate"]

        self.gdf["Current_elec"] = 0

        i = 1
        elec_pop = 0
        total_pop = self.gdf["Calibrated_pop"].sum()

        while elec_pop <= total_pop * elec_rate:
            bool = (self.combined_weight >= i)
            elec_pop = self.gdf.loc[bool, "Calibrated_pop"].sum()

            self.gdf.loc[bool, "Current_elec"] = 1
            i = i - 0.01

        self.i = i

    def final_elec(self):

        elec_rate = self.specs["Elec_rate"]

        self.gdf["Elec_pop_calib"] = self.gdf["Calibrated_pop"]

        i = self.i + 0.01
        total_pop = self.gdf["Calibrated_pop"].sum()
        elec_pop = self.gdf.loc[self.gdf["Current_elec"] == 1, "Calibrated_pop"].sum()
        diff = elec_pop - (total_pop * elec_rate)
        factor = diff / self.gdf["Current_elec"].count()

        while elec_pop > total_pop * elec_rate:

            new_bool = (self.i <= self.combined_weight) & (self.combined_weight <= i)

            self.gdf.loc[new_bool, "Elec_pop_calib"] -= factor
            self.gdf.loc[self.gdf["Elec_pop_calib"] < 0, "Elec_pop_calib"] = 0
            self.gdf.loc[self.gdf["Elec_pop_calib"] == 0, "Current_elec"] = 0
            bool = self.gdf["Current_elec"] == 1

            elec_pop = self.gdf.loc[bool, "Elec_pop_calib"].sum()

            new_bool = bool & new_bool
            if new_bool.sum() == 0:
                i = i + 0.01

        self.gdf.loc[self.gdf["Current_elec"] == 0, "Elec_pop_calib"] = 0

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
                # TODO: test that this works!!!
                # output_path = os.path.join(self.output_directory,
                #                            layer.category,
                #                            layer.name)
                data, meta = layer.get_distance_raster(self.base_layer.path,
                                                       create_raster=False,
                                                       # output_path,
                                                       # self.mask_layer.layer
                                                       )
                data /= 1000  # to convert from meters to km
                self.raster_to_dataframe(data,
                                         name=layer.name + '_dist',
                                         method='read')

    def population_to_dataframe(self, layer=None):
        """
        Takes a population `RasterLayer` as input and extracts the populated points to a GeoDataFrame that is
        saved in `OnSSTOVE.gdf`.
        """
        if not layer:
            if self.base_layer:
                layer = self.base_layer.layer.copy()
                meta = self.base_layer.meta
            else:
                raise ValueError("No population layer was provided as input to the method or in the model base_layer")
        layer[layer == meta['nodata']] = np.nan
        layer[layer == 0] = np.nan
        layer[layer < 1] = np.nan
        self.rows, self.cols = np.where(~np.isnan(layer))
        x, y = rasterio.transform.xy(meta['transform'],
                                     self.rows, self.cols,
                                     offset='center')

        self.gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y),
                                     'Pop': layer[self.rows, self.cols]})
        self.gdf.crs = self.project_crs

    def raster_to_dataframe(self, layer, name=None, method='sample',
                            nodata=np.nan, fill_nodata=None, fill_default_value=0):
        """
        Takes a RasterLayer and a method (sample or read), gets the values from the raster layer using the population points previously extracted and saves the values in a new column of OnSSTOVE.gdf
        """
        if method == 'sample':
            with rasterio.open(layer) as src:
                if src.meta['crs'] != self.gdf.crs:
                    data = sample_raster(layer, self.gdf.to_crs(src.meta['crs']))
                else:
                    data = sample_raster(layer, self.gdf)
        elif method == 'read':
            if fill_nodata:
                if fill_nodata == 'interpolate':
                    mask = layer.copy()
                    mask[mask == nodata] = np.nan
                    if np.isnan(mask[self.rows, self.cols]).sum() > 0:
                        mask[~np.isnan(mask)] = 1
                        rows, cols = np.where(np.isnan(mask) & ~np.isnan(self.base_layer.layer))
                        mask[rows, cols] = 0
                        layer = fillnodata(layer, mask=mask,
                                           max_search_distance=100)
                        layer[(mask == 0) & (np.isnan(layer))] = fill_default_value
                else:
                    raise ValueError('fill_nodata can only be None or "interpolate"')

            data = layer[self.rows, self.cols]
        if name:
            self.gdf[name] = data
        else:
            # TODO: check if changing this to pandas series
            return data

    def calibrate_urban_current_and_future_GHS(self, GHS_path):
        self.raster_to_dataframe(GHS_path, name="IsUrban", method='sample')

        if self.specs["End_year"] > self.specs["Start_year"]:
            population_current = self.specs["Population_end_year"]
            urban_current = self.specs["Urban_start"] * population_current
            rural_current = population_current - urban_current

            population_future = self.specs["Population_end_year"]
            urban_future = self.specs["Urban_end"] * population_future
            rural_future = population_future - urban_future

            rural_growth = (rural_future - rural_current) / (self.specs["End_Year"] - self.specs["Start_Year"])
            urban_growth = (urban_future - urban_current) / (self.specs["End_Year"] - self.specs["Start_Year"])

            self.gdf.loc[self.gdf['IsUrban'] > 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * urban_growth
            self.gdf.loc[self.gdf['IsUrban'] < 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * rural_growth

        self.number_of_households()

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
        self.gdf.loc[self.gdf["IsUrban"] > 20, 'Households'] = self.gdf.loc[
                                                                   self.gdf["IsUrban"] > 20, 'Calibrated_pop'] / \
                                                               self.specs["Urban_HHsize"]

    def get_value_of_time(self):
        """
        Calculates teh value of time based on the minimum wage ($/h) and a
        GIS raster map as wealth index, poverty or GDP
        ----
        0.5 is the upper limit for minimum wage and 0.2 the lower limit
        """
        min_value = np.nanmin(self.gdf['relative_wealth'])
        max_value = np.nanmax(self.gdf['relative_wealth'])
        if 'wage_range' not in self.specs.keys():
            self.specs['wage_range'] = (0.2, 0.5)
        wage_range = (self.specs['wage_range'][1] - self.specs['wage_range'][0])
        norm_layer = (self.gdf['relative_wealth'] - min_value) / (max_value - min_value) * wage_range + \
                     self.specs['wage_range'][0]
        self.gdf['value_of_time'] = norm_layer * self.specs[
            'Minimum_wage'] / 30 / 8  # convert $/months to $/h (8 working hours per day)

    def run(self, technologies='all'):
        print(f'[{self.specs["Country_name"]}] Calculating clean cooking access')
        self.get_clean_cooking_access()
        if self.base_fuel is None:
            print(f'[{self.specs["Country_name"]}] Calculating base fuel properties')
            self.set_base_fuel(self.techs.values())
        if technologies == 'all':
            techs = [tech for tech in self.techs.values()]
        elif isinstance(technologies, list):
            techs = [self.techs[name] for name in technologies]
        else:
            raise ValueError("technologies must be 'all' or a list of strings with the technology names to run.")

        # Based on wealth index, minimum wage and a lower an upper range for cost of oportunity
        print(f'[{self.specs["Country_name"]}] Getting value of time')
        self.get_value_of_time()
        # Loop through each technology and calculate all benefits and costs
        for tech in techs:
            print(f'Calculating health benefits for {tech.name}...')
            tech.morbidity(self)
            tech.mortality(self)
            print(f'Calculating carbon emissions benefits for {tech.name}...')
            tech.carbon_emissions(self)
            print(f'Calculating time saved benefits for {tech.name}...')
            tech.time_saved(self)
            print(f'Calculating costs for {tech.name}...')
            tech.required_energy(self)
            tech.discounted_om(self)
            tech.discounted_inv(self)
            tech.discount_fuel_cost(self)
            tech.salvage(self)
            print(f'Calculating net benefit for {tech.name}...\n')
            if 'w_costs' not in self.specs.keys():
                w_health = 1
                w_environment = 1
                w_social = 1
                w_costs = 1
            else:
                w_health = self.specs['w_health']
                w_environment = self.specs['w_environment']
                w_social = self.specs['w_social']
                w_costs = self.specs['w_costs']
            tech.net_benefit(self, w_health, w_environment, w_social, w_costs)

        print('Getting maximum net benefit technologies...')
        self.maximum_net_benefit(techs)
        print('Extracting indicators...')
        print('    - Lives saved')
        self.extract_lives_saved()
        print('    - Health costs')
        self.extract_health_costs_saved()
        print('    - Time saved')
        self.extract_time_saved()
        print('    - Opportunity cost')
        self.extract_opportunity_cost()
        print('    - Avoided emissions')
        self.extract_reduced_emissions()
        print('    - Avoided emissions costs')
        self.extract_emissions_costs_saved()
        print('    - Investment costs')
        self.extract_investment_costs()
        print('    - Fuel costs')
        self.extract_fuel_costs()
        print('    - OM costs')
        self.extract_om_costs()
        print('    - Salvage value')
        self.extract_salvage()
        print('Done')

    def _get_column_functs(self):
        columns_dict = {column: 'first' for column in self.gdf.columns}
        for column in self.gdf.columns[self.gdf.columns.str.contains('cost|benefit|pop|Pop|Households')]:
            columns_dict[column] = 'sum'
        columns_dict['max_benefit_tech'] = 'first'
        return columns_dict

    def maximum_net_benefit(self, techs):
        net_benefit_cols = [col for col in self.gdf if 'net_benefit_' in col]
        benefits_cols = [col for col in self.gdf if 'benefits_' in col]

        for benefit, net in zip(benefits_cols, net_benefit_cols):
            self.gdf[net + '_temp'] = self.gdf[net]
            self.gdf.loc[self.gdf[benefit] < 0, net + '_temp'] = np.nan

        temps = [col for col in self.gdf if '_temp' in col]
        self.gdf["max_benefit_tech"] = self.gdf[temps].idxmax(axis=1)

        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("net_benefit_", "")
        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("_temp", "")
        self.gdf["maximum_net_benefit"] = self.gdf[temps].max(axis=1)

        gdf = gpd.GeoDataFrame()
        gdf_copy = self.gdf.copy()
        for tech in techs:
            current = (tech.households < gdf_copy['Households']) & \
                      (gdf_copy["max_benefit_tech"] == tech.name)
            dff = gdf_copy.loc[current].copy()
            if current.sum() > 0:
                dff.loc[current, "maximum_net_benefit"] *= tech.factor.loc[current]
                dff.loc[current, f'net_benefit_{tech.name}_temp'] = np.nan

                second_benefit_cols = temps.copy()
                second_benefit_cols.remove(f'net_benefit_{tech.name}_temp')
                second_best = dff.loc[current, second_benefit_cols].idxmax(axis=1)

                second_best.replace(np.nan, 'NaN', inplace=True)
                second_best = second_best.str.replace("net_benefit_", "")
                second_best = second_best.str.replace("_temp", "")
                second_best.replace('NaN', np.nan, inplace=True)

                second_tech_net_benefit = dff.loc[current, second_benefit_cols].max(axis=1) * (1 - tech.factor.loc[current])

                elec_factor = dff['Elec_pop_calib'] / dff['Calibrated_pop']
                dff['max_benefit_tech'] = second_best
                dff['maximum_net_benefit'] = second_tech_net_benefit
                dff['Calibrated_pop'] *= (1 - tech.factor.loc[current])
                dff['Households'] *= (1 - tech.factor.loc[current])

                self.gdf.loc[current, 'Calibrated_pop'] *= tech.factor.loc[current]
                self.gdf.loc[current, 'Households'] *= tech.factor.loc[current]
                if tech.name == 'Electricity':
                    dff['Elec_pop_calib'] *= 0
                #     self.gdf.loc[current, 'Elec_pop_calib'] *= tech.factor.loc[current]
                else:
                    self.gdf.loc[current, 'Elec_pop_calib'] = self.gdf.loc[current, 'Calibrated_pop'] * elec_factor
                    dff['Elec_pop_calib'] = dff['Calibrated_pop'] * elec_factor
                gdf = gdf.append(dff)

        self.gdf = self.gdf.append(gdf)

        for net in net_benefit_cols:
            self.gdf[net + '_temp'] = self.gdf[net]

        temps = [col for col in self.gdf if 'temp' in col]

        for tech in self.gdf["max_benefit_tech"].unique():
            index = self.gdf.loc[self.gdf['max_benefit_tech'] == tech].index
            self.gdf.loc[index, f'net_benefit_{tech}_temp'] = np.nan

        isna = self.gdf["max_benefit_tech"].isna()
        self.gdf.loc[isna, 'max_benefit_tech'] = self.gdf.loc[isna, temps].idxmax(axis=1)
        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("net_benefit_", "")
        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("_temp", "")
        self.gdf.loc[isna, "maximum_net_benefit"] = self.gdf.loc[isna, temps].max(axis=1)

    def add_admin_names(self, admin, column_name):

        if isinstance(admin, str):
            admin = gpd.read_file(admin)

        admin.to_crs(self.gdf.crs, inplace=True)

        self.gdf = gpd.sjoin(self.gdf, admin[[column_name, 'geometry']], how="inner", op='intersects')
        self.gdf.drop('index_right', axis=1, inplace=True)
        self.gdf.sort_index(inplace=True)

    def extract_lives_saved(self):
        self.gdf["deaths_avoided"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].deaths_avoided[row.name], axis=1) #* self.gdf["Households"]

    def extract_health_costs_saved(self):

        self.gdf["health_costs_avoided"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].distributed_morbidity[row.name] +
                        self.techs[row['max_benefit_tech']].distributed_mortality[row.name], axis=1) #* self.gdf["Households"]

    def extract_time_saved(self):
        self.gdf["time_saved"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].total_time_saved[row.name], axis=1) #* self.gdf["Households"]

    def extract_opportunity_cost(self):
        self.gdf["opportunity_cost_gained"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].time_value[row.name], axis=1) #* self.gdf["Households"]

    def extract_reduced_emissions(self):
        # TODO: Fix this

        self.gdf["reduced_emissions"] = self.gdf.apply(
                lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_emissions[row.name], axis=1) #* self.gdf["Households"]
        # except:
        #     self.gdf["reduced_emissions"] = self.gdf.apply(
        #         lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_emissions, axis=1) * self.gdf["Households"]

    def extract_investment_costs(self):

        self.gdf["investment_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_investments[row.name], axis=1) #* self.gdf["Households"]

    def extract_om_costs(self):

        self.gdf["om_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_om_costs[row.name], axis=1) #* self.gdf["Households"]

    def extract_fuel_costs(self):

        self.gdf["fuel_costs"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_fuel_cost[row.name], axis=1) #* self.gdf["Households"]

    def extract_salvage(self):

        self.gdf["salvage_value"] = self.gdf.apply(
            lambda row: self.techs[row['max_benefit_tech']].discounted_salvage_cost[row.name], axis=1) #* self.gdf["Households"]

    def extract_emissions_costs_saved(self):
        # TODO: Fix this

        self.gdf["emissions_costs_saved"] = self.gdf.apply(
                lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_costs[row.name], axis=1) #* self.gdf["Households"]
        # except:
        #     self.gdf["emissions_costs_saved"] = self.gdf.apply(
        #         lambda row: self.techs[row['max_benefit_tech']].decreased_carbon_costs, axis=1) * self.gdf["Households"]


    def gdf_to_csv(self, scenario_name):

        name = os.path.join(self.output_directory, scenario_name)

        pt = self.gdf.to_crs({'init': 'EPSG:3395'})

        pt["X"] = pt["geometry"].x
        pt["Y"] = pt["geometry"].y

        df = pd.DataFrame(pt.drop(columns='geometry'))
        df.to_csv(name)

    def extract_wealth_index(self, wealth_index, file_type="csv", x_column="longitude", y_column="latitude",
                             wealth_column="rwi"):

        if file_type == "csv":
            df = pd.read_csv(wealth_index)

            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_column], df[y_column]))
            gdf.crs = 4326
            gdf.to_crs(self.gdf.crs, inplace=True)

            s1_arr = np.column_stack((self.gdf.centroid.x, self.gdf.centroid.y))
            s2_arr = np.column_stack((gdf.centroid.x, gdf.centroid.y))

            def do_kdtree(combined_x_y_arrays, points):
                mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
                dist, indexes = mytree.query(points)
                return dist, indexes

            results1, results2 = do_kdtree(s2_arr, s1_arr)
            self.gdf["relative_wealth"] = gdf.loc[results2][wealth_column].values

        elif file_type == "point":
            gdf = gpd.read_file(wealth_index)
            gdf.to_crs(self.gdf.crs, inplace=True)

            s1_arr = np.column_stack((self.gdf.centroid.x, self.gdf.centroid.y))
            s2_arr = np.column_stack((gdf.centroid.x, gdf.centroid.y))

            def do_kdtree(combined_x_y_arrays, points):
                mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
                dist, indexes = mytree.query(points)
                return dist, indexes

            results1, results2 = do_kdtree(s2_arr, s1_arr)
            self.gdf["relative_wealth"] = gdf.loc[results2].reset_index()[wealth_column]

        elif file_type == "polygon":
            gdf = gpd.read_file(wealth_index)
            gdf.to_crs(self.gdf.crs, inplace=True)

            gdf.rename(columns={wealth_column: "relative_wealth"}, inplace = True)

            self.gdf = gpd.sjoin(self.gdf, gdf[["relative_wealth", "geometry"]], how="left")
        elif file_type == "raster":
            layer = RasterLayer('Demographics', 'Wealth', layer_path=wealth_index, resample='average')

            layer.align(self.base_layer.path)

            self.raster_to_dataframe(layer.layer, name="relative_wealth", method='read',
                                     nodata=layer.meta['nodata'], fill_nodata='interpolate')
        else:
            raise ValueError("file_type needs to be either csv, raster, polygon or point.")

    def re_name(self, df, labels, variable):
        for value, label in labels.items():
            df[variable] = df[variable].str.replace('_', ' ')
            df.loc[df[variable] == value, variable] = label

    def points_to_raster(self, dff, variable, dtype=rasterio.uint8, nodata=111):
        total_bounds = self.mask_layer.layer['geometry'].total_bounds
        height = round((total_bounds[3] - total_bounds[1]) / 1000)
        width = round((total_bounds[2] - total_bounds[0]) / 1000)
        transform = rasterio.transform.from_bounds(*total_bounds, width, height)
        rasterized = features.rasterize(
            ((g, v) for v, g in zip(dff[variable].values, dff['geometry'].values)),
            out_shape=(height, width),
            transform=transform,
            all_touched=True,
            fill=nodata,
            dtype=dtype)
        meta = dict(driver='GTiff',
                    dtype=rasterized.dtype,
                    count=1,
                    crs=3857,
                    width=width,
                    height=height,
                    transform=transform,
                    nodata=nodata,
                    compress='DEFLATE')
        return rasterized, meta, total_bounds

    def create_layer(self, variable, name=None, labels=None, cmap=None, metric='mean'):
        codes = None
        if self.base_layer is not None:
            layer = np.empty(self.base_layer.layer.shape)
            layer[:] = np.nan
            dff = self.gdf.copy().reset_index(drop=False)
        else:
            dff = self.gdf.copy()

        if isinstance(self.gdf[variable].iloc[0], str):
            if isinstance(labels, dict):
                self.re_name(dff, labels, variable)
            dff[variable] += ' and '
            dff = dff.groupby('index').agg({variable: 'sum', 'geometry': 'first'})
            dff[variable] = [s[0:len(s) - 5] for s in dff[variable]]
            codes = {tech: i for i, tech in enumerate(dff[variable].unique())}
            if isinstance(cmap, dict):
                cmap = {i: cmap[tech] for i, tech in enumerate(dff[variable].unique())}

            if self.rows is not None:
                layer[self.rows, self.cols] = [codes[tech] for tech in dff[variable]]
                meta = self.base_layer.meta
                bounds = self.base_layer.bounds
            else:
                dff['codes'] = [codes[tech] for tech in dff['max_benefit_tech']]
                layer, meta, bounds = self.points_to_raster(dff, 'codes')
        else:
            if metric == 'total':
                dff[variable] = dff[variable] * dff['Households']
                dff = dff.groupby('index').agg({variable: 'sum', 'geometry': 'first'})
            elif metric == 'per_100k':
                dff[variable] = dff[variable] * dff['Households']
                dff = dff.groupby('index').agg({variable: 'sum', 'Calibrated_pop': 'sum',
                                                'geometry': 'first'})
                dff[variable] = dff[variable] * 100000 / dff['Calibrated_pop']
            elif metric == 'per_household':
                dff[variable] = dff[variable] * dff['Households']
                dff = dff.groupby('index').agg({variable: 'sum', 'Households': 'sum',
                                                'geometry': 'first'})
                dff[variable] = dff[variable] / dff['Households']
            else:
                dff = dff.groupby('index').agg({variable: metric, 'geometry': 'first'})
            if self.rows is not None:
                layer[self.rows, self.cols] = dff[variable]
                meta = self.base_layer.meta
                bounds = self.base_layer.bounds
            else:
                layer, meta, bounds = self.points_to_raster(dff, variable, dtype='float32',
                                                            nodata=np.nan)
            variable = variable + '_' + metric
        if name is not None:
            variable = name
        raster = RasterLayer('Output', variable)
        raster.layer = layer
        raster.meta = meta
        # raster.meta.update(nodata=np.nan, dtype='float32')
        raster.bounds = bounds

        return raster, codes, cmap

    def to_raster(self, variable, labels=None, cmap=None, metric='mean'):
        raster, codes, cmap = self.create_layer(variable, labels=labels, cmap=cmap, metric=metric)
        raster.save(os.path.join(self.output_directory, 'Output'))
        print(f'Layer saved in {os.path.join(self.output_directory, "Output", variable + ".tif")}\n')
        if codes and cmap:
            with open(os.path.join(self.output_directory, 'ColorMap.clr'), 'w') as f:
                for label, code in codes.items():
                    r = int(to_rgb(cmap[code])[0] * 255)
                    g = int(to_rgb(cmap[code])[1] * 255)
                    b = int(to_rgb(cmap[code])[2] * 255)
                    f.write(f'{code} {r} {g} {b} 255 {label}\n')

    def plot(self, variable, cmap='viridis', cumulative_count=None, quantiles=None,
             legend_position=(1.05, 1), dpi=150,
             admin_layer=None, title=None, labels=None, legend=True, legend_title='', legend_cols=1, rasterized=True,
             stats=False, stats_position=(1.05, 0.5), stats_fontsize=12, metric='mean',
             save_style=False, classes=5):
        raster, codes, cmap = self.create_layer(variable, labels=labels, cmap=cmap, metric=metric)
        if isinstance(admin_layer, gpd.GeoDataFrame):
            admin_layer = admin_layer
        elif not admin_layer:
            admin_layer = self.mask_layer.layer
        if stats:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=dpi)
            self.add_statistics(ax, stats_position, stats_fontsize)
        else:
            ax = None

        raster.plot(cmap=cmap, cumulative_count=cumulative_count,
                    quantiles=quantiles,
                    categories=codes, legend_position=legend_position,
                    admin_layer=admin_layer, title=title, legend=legend,
                    legend_title=legend_title, legend_cols=legend_cols, rasterized=rasterized,
                    ax=ax)

        if save_style:
            if codes:
                categories = {v: f"{v} = {k}" for k, v in codes.items()}
                quantiles = None
            else:
                categories = False
            raster.save_style(os.path.join(self.output_directory, 'Output'),
                              cmap=cmap, quantiles=quantiles, categories=categories,
                              classes=classes)

    def add_statistics(self, ax, stats_position, fontsize=12):
        summary = self.summary(total=True, pretty=False)
        deaths = TextArea("Deaths avoided", textprops=dict(fontsize=fontsize, color='black'))
        health = TextArea("Health costs avoided", textprops=dict(fontsize=fontsize, color='black'))
        emissions = TextArea("Emissions avoided", textprops=dict(fontsize=fontsize, color='black'))
        time = TextArea("Time saved", textprops=dict(fontsize=fontsize, color='black'))

        texts_vbox = VPacker(children=[deaths, health, emissions, time], pad=0, sep=6)

        deaths_avoided = summary.loc['total', 'deaths_avoided']
        health_costs_avoided = summary.loc['total', 'health_costs_avoided'] / 1000
        reduced_emissions = summary.loc['total', 'reduced_emissions'] / 1000
        time_saved = summary.loc['total', 'time_saved']

        deaths = TextArea(f"{deaths_avoided:,.0f} pp/yr", textprops=dict(fontsize=fontsize, color='black'))
        health = TextArea(f"{health_costs_avoided:,.2f} BUSD", textprops=dict(fontsize=fontsize, color='black'))
        emissions = TextArea(f"{reduced_emissions:,.2f} Bton", textprops=dict(fontsize=fontsize, color='black'))
        time = TextArea(f"{time_saved:,.2f} h/pp.day", textprops=dict(fontsize=fontsize, color='black'))

        values_vbox = VPacker(children=[deaths, health, emissions, time], pad=0, sep=6, align='right')

        hvox = HPacker(children=[texts_vbox, values_vbox], pad=0, sep=6)

        ab = AnnotationBbox(hvox, stats_position,
                            xycoords='axes fraction',
                            box_alignment=(0, 0),
                            pad=0.0,
                            bboxprops=dict(boxstyle='round',
                                           facecolor='#f1f1f1ff',
                                           edgecolor='lightgray'))

        ax.add_artist(ab)

    def to_image(self, variable, name=None, type='png', cmap='viridis', cumulative_count=None,
                 legend_position=(1.05, 1), admin_layer=None, title=None, dpi=300, labels=None, legend=True,
                 legend_title='', legend_cols=1, rasterized=True, stats=False, stats_position=(1.05, 0.5),
                 stats_fontsize=12, metric='mean'):
        raster, codes, cmap = self.create_layer(variable, name=name, labels=labels, cmap=cmap, metric=metric)
        if isinstance(admin_layer, gpd.GeoDataFrame):
            admin_layer = admin_layer
        elif not admin_layer:
            admin_layer = self.mask_layer.layer

        if stats:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=dpi)
            self.add_statistics(ax, stats_position, stats_fontsize)
        else:
            ax = None

        raster.save_image(self.output_directory, type=type, cmap=cmap, cumulative_count=cumulative_count,
                          categories=codes, legend_position=legend_position,
                          admin_layer=admin_layer, title=title, ax=ax, dpi=dpi,
                          legend=legend, legend_title=legend_title, legend_cols=legend_cols, rasterized=rasterized)

    def to_json(self, name):
        self.gdf.to_file(os.path.join(self.output_directory, name), driver='GeoJSON')

    def read_data(self, path):
        self.gdf = gpd.read_file(path)

    def summary(self, total=True, pretty=True, labels=None):
        dff = self.gdf.copy()
        if labels is not None:
            self.re_name(dff, labels, 'max_benefit_tech')
        for attribute in ['maximum_net_benefit', 'deaths_avoided', 'health_costs_avoided', 'time_saved',
                          'opportunity_cost_gained', 'reduced_emissions', 'reduced_emissions', 'emissions_costs_saved',
                          'investment_costs', 'fuel_costs', 'om_costs', 'salvage_value']:
            dff[attribute] *= dff['Households']
        summary = dff.groupby(['max_benefit_tech']).agg({'Calibrated_pop': lambda row: np.nansum(row) / 1000000,
                                                         'maximum_net_benefit': lambda row: np.nansum(row) / 1000000,
                                                         'deaths_avoided': 'sum',
                                                         'health_costs_avoided': lambda row: np.nansum(row) / 1000000,
                                                         'time_saved': 'sum',
                                                         'opportunity_cost_gained': lambda row: np.nansum(row) / 1000000,
                                                         'reduced_emissions': lambda row: np.nansum(row) / 1000000000,
                                                         'emissions_costs_saved': lambda row: np.nansum(row) / 1000000,
                                                         'investment_costs': lambda row: np.nansum(row) / 1000000,
                                                         'fuel_costs': lambda row: np.nansum(row) / 1000000,
                                                         'om_costs': lambda row: np.nansum(row) / 1000000,
                                                         'salvage_value': lambda row: np.nansum(row) / 1000000,
                                                         }).reset_index()
        if total:
            total = summary.sum().rename('total')
            total['max_benefit_tech'] = 'total'
            summary = summary.append(total)

        summary['time_saved'] /= (summary['Calibrated_pop'] * 1000000 * 365)
        if pretty:
            summary.rename(columns={'max_benefit_tech': 'Max benefit technology',
                                    'Calibrated_pop': 'Population (Million)',
                                    'maximum_net_benefit': 'Total net benefit (MUSD)',
                                    'deaths_avoided': 'Total deaths avoided (pp/yr)',
                                    'health_costs_avoided': 'Health costs avoided (MUSD)',
                                    'time_saved': 'hours/pp.day',
                                    'opportunity_cost_gained': 'Opportunity cost avoided (MUSD)',
                                    'reduced_emissions': 'Reduced emissions (Mton CO2eq)',
                                    'emissions_costs_saved': 'Emissions costs saved (MUSD)',
                                    'investment_costs': 'Investment costs (MUSD)',
                                    'fuel_costs': 'Fuel costs (MUSD)',
                                    'om_costs': 'O&M costs (MUSD)',
                                    'salvage_value': 'Salvage value (MUSD)'}, inplace=True)

        return summary

    def plot_split(self, cmap=None, labels=None, save=False, height=1.5, width=2.5):
        df = self.summary(total=False, pretty=False, labels=labels)

        tech_list = df.sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
        ccolor = 'black'

        p = (ggplot(df)
             + geom_col(aes(x='max_benefit_tech', y='Calibrated_pop', fill='max_benefit_tech'))
             + geom_text(aes(y=df['Calibrated_pop'], x='max_benefit_tech',
                             label=df['Calibrated_pop'] / df['Calibrated_pop'].sum()),
                         format_string='{:.0%}',
                         color=ccolor, size=8, va='center', ha='left')
             + ylim(0, df['Calibrated_pop'].max() * 1.15)
             + scale_x_discrete(limits=tech_list)
             + scale_fill_manual(cmap)
             + coord_flip()
             + theme_minimal()
             + theme(legend_position='none')
             + labs(x='', y='Population (Millions)', fill='Cooking technology')
             )
        if save:
            file = os.path.join(self.output_directory, 'tech_split.pdf')
            p.save(file, height=height, width=width)
        else:
            return p

    def plot_costs_benefits(self, cmap=None, labels=None, save=False, height=1.5, width=2.5):
        df = self.summary(total=False, pretty=False, labels=labels)
        df['investment_costs'] -= df['salvage_value']
        df['fuel_costs'] *= -1
        df['investment_costs'] *= -1
        df['om_costs'] *= -1

        dff = df.melt(id_vars=['max_benefit_tech'], value_vars=['health_costs_avoided',
                                                                'investment_costs',
                                                                'fuel_costs',
                                                                'emissions_costs_saved',
                                                                'om_costs',
                                                                'opportunity_cost_gained'])

        dff['variable'] = dff['variable'].str.replace('_', ' ').str.capitalize()

        if cmap is None:
            cmap = {'Health costs avoided': '#542788', 'Investment costs': '#b35806',
                    'Fuel costs': '#f1a340', 'Emissions costs saved': '#998ec3',
                    'Om costs': '#fee0b6', 'Opportunity cost gained': '#d8daeb'}

        tech_list = df.sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
        cat_order = ['Health costs avoided',
                     'Emissions costs saved',
                     'Opportunity cost gained',
                     'Investment costs',
                     'Fuel costs',
                     'Om costs']

        dff['variable'] = pd.Categorical(dff['variable'], categories=cat_order, ordered=True)

        p = (ggplot(dff)
             + geom_col(aes(x='max_benefit_tech', y='value/1000', fill='variable'))
             + scale_x_discrete(limits=tech_list)
             + scale_fill_manual(cmap)
             + coord_flip()
             + theme_minimal()
             + labs(x='', y='Billion USD', fill='Cost / Benefit')
             )

        if save:
            file = os.path.join(self.output_directory, 'benefits_costs.pdf')
            p.save(file, height=height, width=width)
        else:
            return p

    def plot_benefit_distribution(self, type='box', groupby='None', cmap=None, labels=None, save=False, height=1.5, width=2.5):
        if type.lower() == 'box':
            if groupby.lower() == 'isurban':
                df = self.gdf.groupby(['IsUrban', 'max_benefit_tech'])[['health_costs_avoided',
                                                                        'opportunity_cost_gained',
                                                                        'emissions_costs_saved',
                                                                        'salvage_value',
                                                                        'investment_costs',
                                                                        'fuel_costs',
                                                                        'om_costs',
                                                                        'Households',
                                                                        'Calibrated_pop']].sum()
                df.reset_index(inplace=True)
                self.re_name(df, labels, 'max_benefit_tech')
                tech_list = df.groupby('max_benefit_tech')[['Calibrated_pop']].sum()
                tech_list = tech_list.reset_index().sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
                x = 'max_benefit_tech'
            elif groupby.lower() == 'urbanrural':
                df = self.gdf.copy()
                self.re_name(df, labels, 'max_benefit_tech')
                df['Urban'] = df['IsUrban'] > 20
                df['Urban'].replace({True: 'Urban', False: 'Rural'}, inplace=True)
                x = 'Urban'
            else:
                df = self.gdf.copy()
                self.re_name(df, labels, 'max_benefit_tech')
                tech_list = df.groupby('max_benefit_tech')[['Calibrated_pop']].sum()
                tech_list = tech_list.reset_index().sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
                x = 'max_benefit_tech'
            p = (ggplot(df)
                 + geom_boxplot(aes(x=x,
                                    y='(health_costs_avoided + opportunity_cost_gained + emissions_costs_saved' +
                                      ' - investment_costs - fuel_costs - om_costs)',
                                    fill='max_benefit_tech',
                                    color='max_benefit_tech'
                                    ),
                                alpha=0.5, outlier_alpha=0.1, raster=True)
                 + scale_fill_manual(cmap)
                 + scale_color_manual(cmap, guide=False)
                 + coord_flip()
                 + theme_minimal()
                 + labs(y='Net benefit per household (kUSD/yr)', fill='Cooking technology')
                 )
            if groupby.lower() == 'urbanrural':
                p += labs(x='Settlement')
            else:
                p += theme(legend_position="none")
                p += scale_x_discrete(limits=tech_list)
                p += labs(x='')

        elif type.lower() == 'density':
            df = self.gdf.groupby(['IsUrban', 'max_benefit_tech'])[['health_costs_avoided',
                                                                    'opportunity_cost_gained',
                                                                    'emissions_costs_saved',
                                                                    'salvage_value',
                                                                    'investment_costs',
                                                                    'fuel_costs',
                                                                    'om_costs',
                                                                    'Households',
                                                                    'Calibrated_pop']].sum()
            df.reset_index(inplace=True)
            self.re_name(df, labels, 'max_benefit_tech')
            p = (ggplot(df)
                 + geom_density(aes(
                        x='(health_costs_avoided + opportunity_cost_gained + emissions_costs_saved' +
                                      ' + salvage_value - investment_costs - fuel_costs - om_costs)',
                        y=after_stat('count'),
                        fill='max_benefit_tech', color='max_benefit_tech'),
                                alpha=0.1)
                 + scale_fill_manual(cmap, guide=False)
                 + scale_color_manual(cmap)
                 + theme_minimal()
                 + labs(x='Net benefit per household (kUSD/yr)', color='Cooking technology')
                 )
        # compute lower and upper whiskers
        # ylim1 = dff['maximum_net_benefit'].quantile([0.1, 1])/1000

        # scale y limits based on ylim1
        # p = p + coord_flip()

        if save:
            if groupby.lower() not in ['none', '']:
                sufix = f'_{groupby}'
            else:
                sufix = ''
            file = os.path.join(self.output_directory, f'max_benefits_{type}{sufix}.pdf')
            p.save(file, height=height, width=width, dpi=600)
        else:
            return p