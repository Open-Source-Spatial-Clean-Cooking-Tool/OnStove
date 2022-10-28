"""This module contains the model classes of OnStove."""

import os
from typing import Optional, Union, Callable

import dill
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import psycopg2
import scipy.spatial
from copy import copy
from csv import DictReader
from time import time
from matplotlib import cm
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import (TextArea, AnnotationBbox, VPacker, HPacker)
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import transform_bounds
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
    geom_point,
    facet_wrap
)

from onstove.technology import VectorLayer, RasterLayer, Technology, LPG, Biomass, Electricity, Biogas, Charcoal
from onstove.raster import sample_raster


def timeit(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


class DataProcessor:
    """Class containing the methods to process the GIS datasets required for the :class:`MCA` and the :class:`OnStove`
    models.

    Parameters
    ----------
    project_crs: pyproj.CRS or int, optional
        The coordinate system of the project. The value can be anything accepted by
        :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.to_crs`, such as an authority string (eg “EPSG:4326”),
        a WKT string or an EPSG int. If defined all datasets added to the ``DataProcessor`` will be reprojected to
        this crs when calling the :meth:`reproject_layers` method. If not defined then the ``crs`` of the
        :attr:`base_layer` will be used.
    cell_size: tuple of float (width, height), optional
        The desired cell size of the raster layers. It should be defined in the units of the used ``crs``. If defined,
        it will be used to rescale all datasets when calling the :meth:`align` method. If not defined, the
        ``cell_size`` of the :attr:`base_layer` will be used.
    output_directory: str, default 'output'
        A folder path where to save the output datasets.

    Attributes
    ----------
    layers: dict[str, dict[str, 'RasterLayer']]
        All layers added to the ``DataProcessor`` using the :meth:`add_layer` method.
    mask_layer: VectorLayer
        Layer used to mask all datasets when calling the :meth:`mask_layers` method. It is set with the
        :meth:`add_mask_layer` method.
    conn: psycopg2.connect
        Connection to a PostgreSQL database, set with the :meth:`set_postgres` method.
    base_layer:
        RasterLayer to use as template for all raster based data processes. For example, the :meth:`align_layers` uses
        the grid cell of this raster to align all other rasters.
    """

    def __init__(self, project_crs: Optional[Union['pyproj.CRS', int]] = None,
                 cell_size: tuple[float] = None, output_directory: str = 'output'):
        """
        Initializes the class and sets an empty layers dictionaries.
        """
        self.layers = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        self.output_directory = output_directory
        self.mask_layer = None
        self.conn = None
        self.base_layer = None

    def _get_layers(self, layers: dict[str, list[str]]) -> dict[str, dict[str, 'RasterLayer']]:
        """Gets the ``dict(category: dict(name: layer))`` dictionary from the :attr:`layers` attribute.

        Parameters
        ----------
        layers: dict
            Dictionary of ``category``-``list of layer names`` pairs to extract from the :attr:`layers` attribute.

        Returns
        -------
        dict[str, dict[str, 'RasterLayer']]
            Dictionary with the ``category``, ``name``, ``layer`` pairs.
        """
        if layers == 'all':
            _layers = self.layers
        else:
            _layers = {}
            for category, names in layers.items():
                _layers[category] = {}
                for name in names:
                    _layers[category][name] = self.layers[category][name]
        return _layers

    def set_postgres(self, dbname: str, user: str, password: str):
        """
        Wrapper function to set a connection to a PostgreSQL database using the :doc:`psycopg2.connect<psycopg2:module>`
        class.

        It stores the connection into the :attr:`conn` attribute, which can be used to read layers directly from the
        database.

        .. warning::
           The PostgreSQL database connection is only functional for vector layers. Compatibility with raster layers
           will be available in future releases.

        Parameters
        ----------
        dbname: str
            name of the database.
        user: str
            User name of the postgres connection.
        password: str
            Password to authenticate the user.
        """
        self.conn = psycopg2.connect(dbname=dbname,
                                     user=user,
                                     password=password)

    def add_layer(self, category: str, name: str, path: str, layer_type: str, query: str = None,
                  postgres: bool = False, base_layer: bool = False, resample: str = 'nearest',
                  normalization: str = 'MinMax', inverse: bool = False, distance_method: str = 'proximity',
                  distance_limit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                  window: Optional['rasterio.windows.Window'] = None, rescale: bool = False):
        """
        Adds a new layer (type VectorLayer or RasterLayer) to the DataProcessor class

        Parameters
        ----------
        category: str
            Category of the layer. This parameter is useful to group the data into logical categories such as
            "Demographics", "Resources" and "Infrastructure" or "Demand", "Supply" and "Others". This categories are
            particularly relevant for the ``MCA`` analysis.
        name: str
            Name of the dataset.
        path: str
            Path to the layer.
        layer_type: str
            Layer type, `vector` or `raster`.
        query: str, optional
            A query string to filter the data. For more information refer to
            :doc:`pandas:reference/api/pandas.DataFrame.query`.

            .. note::
               Only applicable for the `vector` ``layer_type``.

        postgres: bool, default False
            Whether to use a PostgreSQL database connection to read the layer from. The connection needs be already
            created and stored in the :attr:`conn` attribute using the :meth:`set_postgres` method.
        base_layer: bool, default False
            Whether the layer should be used as a :attr:`base_layer` for the data processing.
        resample: str, default 'nearest'
            Sets the default method to use when resampling the dataset. Resampling occurs when changing the grid cell size
            of the raster, thus the values of the cells need to be readjusted to reflect the new cell size. Several
            sampling methods can be used, and which one to use is dependent on the nature of the data. For a list of the
            accepted methods refer to :doc:`rasterio.enums.Resampling<rasterio:api/rasterio.enums>`.

            .. seealso::
               :class:`RasterLayer`

        normalization: str, 'MinMax'
            Sets the default normalization method to use when calling the :meth:`RasterLayer.normalize`. This is relevant
            to calculate the
            :attr:`demand_index<onstove.MCA.demand_index>`,
            :attr:`supply_index<onstove.MCA.supply_index>`,
            :attr:`clean_cooking_index<onstove.MCA.clean_cooking_index>` and
            :attr:`assistance_need_index<onstove.MCA.assistance_need_index>` of the ``MCA`` model.

            .. note::
               If the ``layer_type`` is `vector`, this parameters will be passed to any raster dataset created from
               the vector layer, for example see :attr:`VectorLayer.distance_raster`.

        inverse: bool, default False
            Sets the default mode for the normalization algorithm (see :meth:`RasterLayer.normalize`).

            .. note::
               If the ``layer_type`` is `vector`, this parameters will be passed to any raster dataset created from
               the vector layer, for example see :attr:`VectorLayer.distance_raster`.

        distance_method: str, default 'proximity'
            Sets the default distance algorithm to use when calling the :meth:`get_distance_raster` method for the
            layer, see :class:`VectorLayer` and :class:`RasterLayer`.
        distance_limit: Callable object (function or lambda function) with a numpy array as input, optional
            Defines a distance limit or range to consider when calculating the distance raster, see
            :class:`VectorLayer` and :class:`RasterLayer`.
        window: rasterio.windows.Window or gpd.GeoDataFrame
            A window or bounding box to read in the data if ``layer_type`` is `raster` or `vector` respectively.
            See the ``window`` parameter of :class:`RasterLayer` and the ``bbox`` parameter of :class:`VectorLayer`.
        rescale: bool, default False
            Sets the default value for the ``rescale`` attribute. See the ``rescale`` parameter of :class:`RasterLayer`.
        """
        if layer_type == 'vector':
            if postgres:
                layer = VectorLayer(category, name, path, conn=self.conn,
                                    normalization=normalization,
                                    distance_method=distance_method,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)
            else:
                if window:
                    window = self.mask_layer.data
                layer = VectorLayer(category, name, path,
                                    normalization=normalization,
                                    distance_method=distance_method,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query, bbox=window)

        elif layer_type == 'raster':
            if window:
                with rasterio.open(path) as src:
                    src_crs = src.meta['crs']
                if src_crs != self.mask_layer.data.crs:
                    bounds = transform_bounds(self.mask_layer.data.crs, src_crs, *self.mask_layer.bounds)
                window = bounds
            layer = RasterLayer(category, name, path,
                                normalization=normalization, inverse=inverse,
                                distance_method=distance_method, resample=resample,
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

    def add_mask_layer(self, category: str, name: str, path: str,
                       query: str = None, postgres: bool = False):
        """
        Adds a vector layer to self.mask_layer, which will be used to mask all
        other layers into is boundaries

        Parameters
        ----------
        category: str
            Category name of the dataset.
        name: str
            name of the dataset.
        path: str
            The relative path to the datafile. This file can be of any type that is accepted by
            :doc:`geopandas:docs/reference/api/geopandas.read_file`.
        query: str, optional
            A query string to filter the data. For more information refer to
            :doc:`pandas:reference/api/pandas.DataFrame.query`.
        postgres: bool, default False
            Whether to use a PostgreSQL database connection to read the layer from. The connection needs be already
            created and stored in the :attr:`conn` attribute using the :meth:`set_postgres` method.
        """
        if postgres:
            self.mask_layer = VectorLayer(category, name, path, self.conn, query)
        else:
            self.mask_layer = VectorLayer(category, name, path, query=query)

        if self.mask_layer.data.crs != self.project_crs:
            output_path = os.path.join(self.output_directory, category, name)
            self.mask_layer.reproject(self.project_crs, output_path)

        self.mask_layer.save(os.path.join(self.output_directory, self.mask_layer.category, self.mask_layer.name))

    def mask_layers(self, datasets: dict[str, list[str]] = 'all'):
        """
        Uses the a mask layer in ``self.mask_layer`` to mask all other layers to its boundaries.

        Parameters
        ----------
        datasets: dictionary of ``category``-``list of layer names`` pairs, default 'all'
            Specifies which dataset(s) to clip.

        See also
        ----------
        add_mask_layer
        """
        if not isinstance(self.mask_layer, VectorLayer):
            raise Exception('The `mask_layer` attribute is empty, please first ' + \
                            'add a mask layer using the `.add_mask_layer` method.')
        datasets = self._get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                if name != self.base_layer.name:
                    all_touched = True
                else:
                    all_touched = False
                if isinstance(layer, RasterLayer):
                    layer.mask(self.mask_layer, output_path, all_touched=all_touched)
                elif isinstance(layer, VectorLayer):
                    layer.mask(self.mask_layer, output_path)

                if isinstance(layer.friction, RasterLayer):
                    layer.friction.mask(self.mask_layer, output_path)
                if isinstance(layer.distance_raster, RasterLayer):
                    layer.distance_raster.mask(self.mask_layer, output_path)

    def align_layers(self, datasets: dict[str, list[str]] = 'all'):
        """
        Ensures that the coordinate system and resolution of the raster is the same as the base layer

        Parameters
        ----------
        datasets: dictionary of ``category``-``list of layer names`` pairs, default 'all'
            Specifies which dataset(s) to align.

        See also
        ----------
        add_layer
        """

        datasets = self._get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                if isinstance(layer, VectorLayer):
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(base_layer=self.base_layer, output_path=output_path)
                else:
                    if name != self.base_layer.name:
                        layer.align(base_layer=self.base_layer, output_path=output_path)
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(base_layer=self.base_layer, output_path=output_path)

    def reproject_layers(self, datasets: dict[str, list[str]] = 'all'):
        """
        Reprojects the layers specified by the user.

        Parameters
        ----------
        datasets: dictionary of ``category``-``list of layer names`` pairs, default 'all'
            Specifies which dataset(s) to reproject.

        See also
        --------
        RasterLayer.reproject
        VectorLayer.reproject
        """
        datasets = self._get_layers(datasets)
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
        datasets = self._get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.get_distance_raster(self.base_layer.path,
                                          output_path, self.mask_layer.data)
                # if isinstance(layer.friction, RasterLayer):
                #     layer.friction.get_distance_raster(self.base_layer.path,
                #                                        output_path, self.mask_layer.layer)

    def normalize_rasters(self, datasets='all', buffer=False):
        """
        Goes through all layer and call their `.normalize` method
        """
        datasets = self._get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.distance_raster.normalize(output_path, self.mask_layer.data, buffer=buffer)

    def save_datasets(self, datasets='all'):
        """
        Saves all layers that have not been previously saved
        """
        datasets = self._get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                layer.save(output_path)

    def to_pickle(self, name):
        """Saves the model as a pickle."""
        self.conn = None
        with open(os.path.join(self.output_directory, name), "wb") as f:
            dill.dump(self, f)

    @classmethod
    def read_model(cls, path):
        """Reads a model from a pickle"""
        with open(path, "rb") as f:
            model = dill.load(f)
        return model


class MCA(DataProcessor):
    """The ``MCA`` class is used to conduct a spatial Multicriteria Analysis in order to prioritize areas of action for
    clean cooking access.

    The MCA model is based in the methods of the Energy Access Explorer (EAE) and the Clean Cooking Explorer (CCE). It
    focuses on identifying potential areas where clean cooking can be quickly adopted, areas where markets for
    clean cooking technologies can be expanded or areas in need of financial assistance or lack of infrastructure.
    In brief, it identifies priority areas of action from the user perspective.

    .. note::
       The ``OnStove`` class inherits all functionalities from the :class:`DataProcessor` class.

    Parameters
    ----------
    **kwargs: dict of parameters
        Parameters from the :class:`DataProcessor` parent class.

    Attributes
    ----------
    demand_index
    supply_index
    clean_cooking_index
    assistance_need_index
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.demand_index = None
        self.supply_index = None
        self.clean_cooking_index = None
        self.assistance_need_index = None

    @property
    def demand_index(self):
        """The Demand Index highlights the potential demand for clean cooking in different parts of the study area. It
        shows where demand is comparatively higher or lower.

        The demand index is generated by calling the :meth:`get_demand_index` method, which normalizes and weights all
        relevant demand datasets and combines them.

        See also
        --------
        get_demand_index
        supply_index
        get_supply_index
        clean_cooking_index
        get_clean_cooking_index
        assistance_need_index
        get_assistance_need_index
        """
        if self._demand_index is None:
            raise ValueError('No `demand_index` was found, please calculate a demand index by calling the '
                             '`get_demand_index()` method with the relevant list of `datasets`.')
        return self._demand_index

    @demand_index.setter
    def demand_index(self, raster):
        if isinstance(raster, RasterLayer):
            self._demand_index = raster
        elif raster is None:
            self._demand_index = None
        else:
            raise ValueError('The demand index needs to be of class `RasterLayer`.')

    @property
    def supply_index(self):
        """The Supply Index highlights the potential for clean cooking supply in different parts of the study area. It
        shows where supply is comparatively better or worst.

        This index is generated by calling the :meth:`get_supply_index` method, which normalizes and weights all
        relevant demand datasets and combines them. The supply index can show where the supply potential is better
        for different types of stoves, e.g. biogas, LPG or electricity and Improved Cook Stoves (ICS), or where supply
        is more easily accessible.

        See also
        --------
        get_supply_index
        demand_index
        get_demand_index
        clean_cooking_index
        get_clean_cooking_index
        assistance_need_index
        get_assistance_need_index
        """
        if self._supply_index is None:
            raise ValueError('No `supply_index` was found, please calculate a supply index by calling the '
                             '`get_supply_index()` method with the relevant list of `datasets`.')
        return self._supply_index

    @supply_index.setter
    def supply_index(self, raster):
        if isinstance(raster, RasterLayer):
            self._supply_index = raster
        if raster is None:
            self._supply_index = None
        else:
            raise ValueError('The supply index needs to be of class `RasterLayer`.')

    @property
    def clean_cooking_index(self):
        """The Clean Cooking Index measures where demand and supply are simultaneously higher.

        This index is generated by calling the :meth:`get_clean_cooking_index` method, which produces an aggregated
        measure of the :attr:`demand_index` and the :attr:`supply_index`. Areas with high demand and high supply get
        a higher clean cooking index.

        See also
        --------
        get_clean_cooking_index
        demand_index
        get_demand_index
        supply_index
        get_supply_index
        assistance_need_index
        get_assistance_need_index
        """
        if self._clean_cooking_index is None:
            raise ValueError('No `clean_cooking_index` was found, please calculate a clean cooking index by calling '
                             'the `get_clean_cooking_index()` method with the relevant list of `datasets`.')
        return self._clean_cooking_index

    @clean_cooking_index.setter
    def clean_cooking_index(self, raster):
        if isinstance(raster, RasterLayer):
            self._clean_cooking_index = raster
        if raster is None:
            self._clean_cooking_index = None
        else:
            raise ValueError('The clean cooking index needs to be of class `RasterLayer`.')

    @property
    def assistance_need_index(self):
        """The Clean Cooking Index measures where demand and supply are simultaneously higher.

        This index is generated by calling the :meth:`get_clean_cooking_index` method, which produces an aggregated
        measure of the :attr:`demand_index` and the :attr:`supply_index`. Areas with high demand and high supply get
        a higher clean cooking index.

        See also
        --------
        get_clean_cooking_index
        demand_index
        get_demand_index
        supply_index
        get_supply_index
        assistance_need_index
        get_assistance_need_index
        """
        if self._assistance_need_index is None:
            raise ValueError('No `assistance_need_index` was found, please calculate an assistance need index by '
                             'calling the `get_assistance_need_index()` method with the relevant list of `datasets`.')
        return self._assistance_need_index

    @assistance_need_index.setter
    def assistance_need_index(self, raster):
        if isinstance(raster, RasterLayer):
            self._assistance_need_index = raster
        if raster is None:
            self._assistance_need_index = None
        else:
            raise ValueError('The assistance need index needs to be of class `RasterLayer`.')

    @staticmethod
    def index(layers):
        data = {}
        for k, i in layers.items():
            data.update(i)
        weights = []
        rasters = []
        for name, layer in data.items():
            rasters.append(layer.weight * layer.distance_raster.normalized.data)
            weights.append(layer.weight)

        return sum(rasters) / sum(weights)

    def get_demand_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self._get_layers(datasets)
        layer = RasterLayer('Indexes', 'Demand_index', normalization='MinMax')
        layer.data = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Demand Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.data, buffer=buffer)
        layer.normalized.name = 'Demand Index'
        self.demand_index = layer.normalized

    def get_supply_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self._get_layers(datasets)
        layer = RasterLayer('Indexes', 'Supply index', normalization='MinMax')
        layer.data = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Supply Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.data, buffer=buffer)
        layer.normalized.name = 'Supply Index'
        self.supply_index = layer.normalized

    def get_clean_cooking_index(self, demand_weight=1, supply_weight=1, buffer=False):
        layer = RasterLayer('Indexes', 'Clean Cooking Potential Index', normalization='MinMax')
        layer.data = (demand_weight * self.demand_index.data + supply_weight * self.supply_index.data) / \
                     (demand_weight + supply_weight)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Clean Cooking Potential Index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.data, buffer=buffer)
        layer.normalized.name = 'Clean Cooking Potential Index'
        self.clean_cooking_index = layer.normalized

    def get_assistance_need_index(self, datasets='all', buffer=False):
        self.normalize_rasters(datasets=datasets, buffer=buffer)
        datasets = self._get_layers(datasets)
        layer = RasterLayer('Indexes', 'Assistance need index', normalization='MinMax')
        layer.data = self.index(datasets)
        layer.meta = self.base_layer.meta
        output_path = os.path.join(self.output_directory,
                                   'Indexes', 'Assistance need index')
        os.makedirs(output_path, exist_ok=True)
        layer.normalize(output_path, self.mask_layer.data, buffer=buffer)
        layer.normalized.name = 'Assistance need index'
        self.assistance_need_index = layer.normalized

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
            data = self.clean_cooking_index.data
        elif index.lower() in 'assistance need index':
            data = self.assistance_need_index.data
        elif index.lower() in 'supply index':
            data = self.supply_index.data

        for level in [0.2, 0.4, 0.6, 0.8, 1]:
            levels.append(np.where(
                (data >= (level - 0.2)) & (data < level)))

        share = []
        for level in levels:
            value = np.nansum(self.layers[layer[0]][layer[1]].data[level])
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


class OnStove(DataProcessor):
    """The ``OnStove`` class is used to perform a geospatial cost-benefit analysis on clean cooking access.

    OnStove determines the net-benefits of cooking with different stoves across an area with regards to capital, fuel,
    and operation and maintenance costs, as well as benefits from reduced morbidity, reduced mortality, time saved and
    emissions avoided. The model identifies the stove that can provide cooking in each settlement achieving the highest
    net-benefit.

    .. note::
       The ``OnStove`` class inherits all functionalities from the :class:`DataProcessor` class.

    Parameters
    ----------
    project_crs: pyproj.CRS or int, optional
        See the :class:`DataProcessor` class for details.
    cell_size: float, optional
        See the :class:`DataProcessor` class for details.
    output_directory: str, default 'output'
        A folder path where to save the output datasets.

    Attributes
    ----------
    rows
    cols
    specs
    techs
    base_fuel
    energy_per_meal
    gwp
    clean_cooking_access_u
    clean_cooking_access_r
    electrified_weight
    """

    def __init__(self, project_crs: Optional[Union['pyproj.CRS', int]] = None,
                 cell_size: float = None, output_directory: str = 'output'):
        """
        Initializes the class and sets an empty layers dictionaries.
        """
        super().__init__(project_crs, cell_size, output_directory)
        self.rows = None
        self.cols = None
        self.specs = None
        self.techs = {}
        self.base_fuel = None
        self.i = {}
        self.energy_per_meal = 3.64  # MJ
        # TODO: remove from here and make it an input in the specs file
        self.gwp = {'co2': 1, 'ch4': 25, 'n2o': 298, 'co': 2, 'bc': 900, 'oc': -46}
        self.clean_cooking_access_u = None
        self.clean_cooking_access_r = None
        self.electrified_weight = None

    def read_scenario_data(self, path_to_config: str, delimiter=','):
        """Reads the scenario data into a dictionary
        """
        config = {}
        with open(path_to_config, 'r') as csvfile:
            reader = DictReader(csvfile, delimiter=delimiter)
            config_file = list(reader)
            for row in config_file:
                if row['Value'] is not None:
                    if row['data_type'] == 'int':
                        config[row['Param'].lower()] = int(row['Value'])
                    elif row['data_type'] == 'float':
                        config[row['Param'].lower()] = float(row['Value'])
                    elif row['data_type'] == 'string':
                        config[row['Param'].lower()] = str(row['Value'])
                    elif row['data_type'] == 'bool':
                        config[row['Param'].lower()] = str(row['Value']).lower() in ['true', 't', 'yes', 'y', '1']
                    else:
                        raise ValueError("Config file data type not recognised.")
        if self.specs is None:
            self.specs = config
        else:
            self.specs.update(config)

        self.check_scenario_data()


    def check_scenario_data(self):
        """This function checks goes through all rows without default values needed in the socio-economic specification
        file to check whether they are included or not. If they are included nothing happens, otherwise a ValueError will
        be raised.
        """

        rows = ['country_name', 'country_code', 'population_start_year', 'population_end_year', 'urban_start',
                'urban_end', 'elec_rate', 'rural_elec_rate', 'urban_elec_rate', 'mort_copd', 'mort_ihd', 'mort_lc',
                'mort_alri', 'morb_copd','morb_ihd', 'morb_lc', 'morb_alri', 'rural_hh_size', 'urban_hh_size',
                'mort_stroke', 'morb_stroke', 'fnrb','coi_alri', 'coi_copd', 'coi_ihd', 'coi_lc', 'coi_stroke',
                'cost_of_carbon_emissions', 'minimum_wage', 'vsl']

        for row in rows:
            if row not in self.specs:
                raise ValueError("The socio-economic file has to include the " + row + " row")

    def techshare_sumtoone(self):
        """This function checks if the sum of shares in the technology dictionary is 1.0.

        If it is not, it will adjust the shares to make the sum 1.0.

        Parameters
        ----------
        techshare_dict : dictionary
            The original dictionary of technology shares

        Returns
        -------
        techshare_dict : dictionary
            The updated dictionary of technology shares
        """
        sharesumrural = sum(item['current_share_rural'] for item in self.techs.values())

        if sharesumrural != 1:
            for item in self.techs.values():
                item.current_share_rural = item.current_share_rural / sharesumrural

        sharesumurban = sum(item['current_share_urban'] for item in self.techs.values())

        if sharesumurban != 1:
            for item in self.techs.values():
                item.current_share_urban = item.current_share_urban / sharesumurban

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
            share = tech.current_share_rural + tech.current_share_urban
            if (share > 0) or tech.is_base:
                tech.is_base = True
                base_fuels.append(tech)
        if len(base_fuels) == 1:
            self.base_fuel = copy(base_fuels[0])
            self.base_fuel.carb(self)
            self.base_fuel.total_time(self)
            self.base_fuel.required_energy(self)
            self.base_fuel.adjusted_pm25()
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

                tech.discounted_inv(self, relative=False)
                base_fuel.tech_life += tech.tech_life * current_share
                base_fuel.discounted_investments += tech.discounted_investments * current_share

                tech.adjusted_pm25()
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
                        elif 'pellets' in row['Fuel'].lower():
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
        """Calculates the clean cooking access in rural and urban settlements.

        It uses the :attr:`Technology.current_share_urban` and :attr:`Technology.current_share_rural` attributes,
        from each technology class, in combination with the :attr:`Technology.is_clean` attribute to get the current
        clean cooking access as a percentage.

        Returns
        -------
        clean_cooking_access_u: float
            Stores the clean cooking access percentage in urban settlements in the :attr:`clean_cooking_access_u`
            attribute.
        clean_cooking_access_r: float
            Stores the clean cooking access percentage in rural settlements in the :attr:`clean_cooking_access_r`
            attribute.
        """
        clean_cooking_access_u = 0
        clean_cooking_access_r = 0
        for tech in self.techs.values():
            if tech.is_clean:
                clean_cooking_access_u += tech.current_share_urban
                clean_cooking_access_r += tech.current_share_rural
        self.clean_cooking_access_u = clean_cooking_access_u
        self.clean_cooking_access_r = clean_cooking_access_r

    def normalize(self, column: str, inverse: bool = False):
        """Uses the MinMax method to normalize the data from a column of the :attr:`gdf` GeoDataFrame.

        Parameters
        ----------
        column: str
            Name of the column of the :attr:`gdf` to create the normalized data from.
        inverse: bool, default False
            Whether to invert the range in the normalized data.

        Returns
        -------
        pd.Series
            The normalized pd.Series.
        """
        if inverse:
            normalized = (self.gdf[column].max() - self.gdf[column]) / (self.gdf[column].max() - self.gdf[column].min())
        else:
            normalized = (self.gdf[column] - self.gdf[column].min()) / (self.gdf[column].max() - self.gdf[column].min())

        return normalized

    @property
    def electrified_weight(self):
        """Spatial weighted average factor used to calibrate the current electrified population.

        It uses the night time lights, population count and distance to electricity infrastructure (this can be either
        transformers, medium voltage lines or high voltage lines in that order of preference) in combination with
        user-defined weights to calculate the factor. This factor serves to provide a "probability" for each settlement
        to be electrified, which will be used in the calibration of electrified population.

        See also
        --------
        current_elec
        final_elec
        """
        if self._electrified_weight is None:
            if "Transformers_dist" in self.gdf.columns:
                self.gdf["Elec_dist"] = self.gdf["Transformers_dist"]
            elif "MV_lines_dist" in self.gdf.columns:
                self.gdf["Elec_dist"] = self.gdf["MV_lines_dist"]
            else:
                self.gdf["Elec_dist"] = self.gdf["HV_lines_dist"]

            elec_dist = self.normalize("Elec_dist", inverse=True)
            ntl = self.normalize("Night_lights")
            pop = self.normalize("Calibrated_pop")

            weight_sum = elec_dist * self.specs["infra_weight"] + pop * self.specs["pop_weight"] + \
                         ntl * self.specs["NTL_weight"]
            weights = self.specs["infra_weight"] + self.specs["pop_weight"] + self.specs["NTL_weight"]

            self.electrified_weight = weight_sum / weights
        return self._electrified_weight

    @electrified_weight.setter
    def electrified_weight(self, value):
        self._electrified_weight = value

    def current_elec(self):
        """Calculates a binary variable that defines which settlements are at least partially electrified.

        It uses the electrification rate provided by the user in the :attr:`specs` file (named as ``Elec_rate``) and
        the :attr:`electrified_weight` to make the calibration. The binary variable is saved as a column of the
        GeoDataFrame :attr:`gdf` with the name of ``Current_elec``.

        See also
        --------
        electrified_weight
        final_elec
        read_scenario_data
        specs
        """
        elec_rate = self.specs["Elec_rate"]

        self.gdf["Current_elec"] = 0

        i = 1
        elec_pop = 0
        total_pop = self.gdf["Calibrated_pop"].sum()

        while elec_pop <= total_pop * elec_rate:
            bool = (self.electrified_weight >= i)
            elec_pop = self.gdf.loc[bool, "Calibrated_pop"].sum()

            self.gdf.loc[bool, "Current_elec"] = 1
            i = i - 0.01

        self.i = i

    def final_elec(self):
        """Calibrates the electrified population within each cell.

        This is a "fine-tuning" of the electrified population. It uses the ``Current_elec`` column of the :attr:`gdf`
        GeoDataFrame (calculated using the :meth:`current_elec` method) and the ``Calibrated_pop`` column (calculated
        using the :meth:`calibrate_current_pop` method) to get the population that is electrified within each
        electrified settlement, according to the ``Elec_rate`` provided by the user (stored in :attr:`specs`).

        See also
        --------
        electrified_weight
        current_elec
        read_scenario_data
        specs
        """
        elec_rate = self.specs["Elec_rate"]

        self.gdf["Elec_pop_calib"] = self.gdf["Calibrated_pop"]

        i = self.i + 0.01
        total_pop = self.gdf["Calibrated_pop"].sum()
        elec_pop = self.gdf.loc[self.gdf["Current_elec"] == 1, "Calibrated_pop"].sum()
        diff = elec_pop - (total_pop * elec_rate)
        factor = diff / self.gdf["Current_elec"].count()

        while elec_pop > total_pop * elec_rate:

            new_bool = (self.i <= self.electrified_weight) & (self.electrified_weight <= i)

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
        """Calibrates the spatial population in each cell according to the user defined population in the start year
        (``Population_start_year`` in the :attr:`specs` dictionary) and saves it in the ``Calibrated_pop`` column of
        the main GeoDataFrame (:attr:`gdf`).

        See also
        --------
        read_scenario_data
        specs
        gdf
        """
        total_gis_pop = self.gdf["Pop"].sum()
        calibration_factor = self.specs["Population_start_year"] / total_gis_pop

        self.gdf["Calibrated_pop"] = self.gdf["Pop"] * calibration_factor

    def distance_to_electricity(self, hv_lines: VectorLayer = None, mv_lines: VectorLayer = None,
                                transformers: VectorLayer = None):
        """ Calculates the distance to electricity infrastructure.

        It calls the :meth:`VectorLayer.get_distance_raster` method for the high voltage, medium voltage and
        transformers datasets (if available) and converts the output to a column in the main GeoDataFrame (:attr:`gdf`).

        .. warning::
           The ``name`` attribute of the input datasets must be `HV_lines`, `MV_lines` and `Transformers`. This
           naming convention may change in future releases.

        Parameters
        ----------
        hv_lines: VectorLayer
            High voltage lines dataset.
        mv_lines: VectorLayer
            Medium voltage lines dataset.
        transformers: VectorLayer
            Transformers dataset.
        """
        if (not hv_lines) and (not mv_lines) and (not transformers):
            raise ValueError("You MUST provide at least one of the following datasets: hv_lines, mv_lines or "
                             "transformers.")

        for layer in [hv_lines, mv_lines, transformers]:
            if layer:
                layer.get_distance_raster(raster=self.base_layer)
                layer.distance_raster.data /= 1000  # to convert from meters to km
                self.raster_to_dataframe(layer.distance_raster,
                                         name=layer.name + '_dist',
                                         method='read')

    def population_to_dataframe(self, layer: Optional[RasterLayer] = None):
        """
        Takes a population `RasterLayer` as input and extracts the populated points to the main GeoDataFrame saved in
        the :attr:`gdf` attribute.

        Parameters
        ----------
        layer: RasterLayer
            The raster layer containing the population count data. If not defined, then the :attr:`base_layer` dataset
            will be used. If ``layer`` is not provided and :attr:`base_layer` is None, then an error will be raised.
        """

        if isinstance(layer, RasterLayer):
            data = layer.data.copy()
            meta = layer.meta
        else:
            if self.base_layer:
                data = self.base_layer.data.copy()
                meta = self.base_layer.meta
            else:
                raise ValueError("No population layer was provided as input to the method or in the model base_layer")

        data[data == meta['nodata']] = np.nan
        data[data == 0] = np.nan
        data[data < 1] = np.nan
        self.rows, self.cols = np.where(~np.isnan(data))
        x, y = rasterio.transform.xy(meta['transform'],
                                     self.rows, self.cols,
                                     offset='center')

        self.gdf = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(x, y),
                                     'Pop': data[self.rows, self.cols]})
        self.gdf.crs = self.project_crs

    def raster_to_dataframe(self, layer: Union[RasterLayer, str], name: Optional[str] = None, method: str = 'sample',
                            fill_nodata_method: Optional[str] = None,
                            fill_default_value: Union[float, int] = 0):
        """
        Takes a :class:`RasterLayer` and a method (``sample`` or ``read``) and extracts the values from the raster
        layer to the main GeoDataFrame (:attr:`gdf`).

        It uses the coordinates of the population points (previously extracted with the :meth:`population_to_dataframe`
        method) to either sample the values from the dataset (if ``sample`` is used) or read the :attr:`rows` and
        :attr:`cols` from the array (if ``read`` is used). The further requires that the raster dataset is aligned with
        the used population layer.

        Parameters
        ----------
        layer: RasterLayer or path to the raster
            Raster layer to extract values from. If the method ``sample`` is used, the layer must be provided as the
            path to the raster file.
        name: str, optional
            Name to use for the column of the extracted data in the :attr:`gdf`.
        method: str, default 'sample'
            Method to use when extracting the data. If ``sample``, the values will be sampled using the coordinates of
            the point of the GeoDataFrame (:attr:`gdf`), which have been previously defined by the population layer. if
            ``read``, the values are extracted using the the :attr:`rows` and :attr:`cols` attributes, which have been
            previously extracted using the population layer.
        fill_nodata_method: str, optional
            Method to use to fill the no data. Currently only ``interpolate`` is available.
        fill_default_value: float or int, default 0
            Default value to use to fill in the no data. This will be used for cells that fall outside the search
            radius (currently of 100) if the ``interpolate`` method is selected, and for all the nodata values if
            ``None`` is used as method.
        """
        data = None
        if method == 'sample':
            with rasterio.open(layer) as src:
                if src.meta['crs'] != self.gdf.crs:
                    data = sample_raster(layer, self.gdf.to_crs(src.meta['crs']))
                else:
                    data = sample_raster(layer, self.gdf)
        elif method == 'read':
            if 'nodata' in layer.meta.keys():
                nodata = layer.meta['nodata']
            else:
                nodata = np.nan
            layer = layer.data.copy()
            if fill_nodata_method:
                mask = layer.copy()
                mask[mask == nodata] = np.nan
                if np.isnan(mask[self.rows, self.cols]).sum() > 0:
                    mask[~np.isnan(mask)] = 1
                    base = self.base_layer.data.copy()
                    base[base == self.base_layer.meta['nodata']] = np.nan
                    rows, cols = np.where(np.isnan(mask) & ~np.isnan(base))
                    mask[rows, cols] = 0
                    if fill_nodata_method == 'interpolate':
                        layer = fillnodata(layer, mask=mask,
                                           max_search_distance=100)
                    elif fill_nodata_method is not None:
                        raise ValueError('fill_nodata can only be None or "interpolate"')
                    layer[(mask == 0) & (np.isnan(layer))] = fill_default_value

            data = layer[self.rows, self.cols]
        if name:
            self.gdf[name] = data
        else:
            return data

    def calibrate_urban_rural_split(self, GHS_path: str):
        """Calibrates the urban rural split using spatial data from the
        `GHS SMOD dataset <https://ghsl.jrc.ec.europa.eu/download.php?ds=smod>`_.

        The GHS dataset is used to determine which settlements are urban and which are rural in the analysis. Areas
        that have coding of either 30, 23 or 22 are considered urban, while the rest are rural. It saves the
        binary (1 for urban, 0 for rural) classification as a column in the main GeoDataFrame (:attr:`gdf`) with the
        name of ``IsUrban``.

        Parameters
        ----------
        GHS_path: str
            Path to the GHS dataset
        """
        self.raster_to_dataframe(GHS_path, name="IsUrban", method='sample')

        if self.specs["End_year"] > self.specs["Start_year"]:
            population_current = self.specs["Population_start_year"]
            urban_current = self.specs["Urban_start"] * population_current
            rural_current = population_current - urban_current

            population_future = self.specs["Population_end_year"]
            urban_future = self.specs["Urban_end"] * population_future
            rural_future = population_future - urban_future

            rural_growth = (rural_future - rural_current) / (self.specs["End_year"] - self.specs["Start_year"])
            urban_growth = (urban_future - urban_current) / (self.specs["End_year"] - self.specs["Start_year"])

            self.gdf.loc[self.gdf['IsUrban'] > 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * urban_growth
            self.gdf.loc[self.gdf['IsUrban'] < 20, 'Pop_future'] = self.gdf["Calibrated_pop"] * rural_growth

        self.number_of_households()

    def calibrate_urban_manual(self):
        """Calibrates the urban rural split based on population density.

        It uses the ``Calibrated_pop`` column of the main GeoDataFrame (:attr:`gdf`) and the current national urban
        split defined in :attr:`specs`, to classify the settlements until the total urban population sum matches the
        defined split.
        """
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
        """Calculates the number of households withing each cell based on their urban/rural classification and a
        defined household size.

        It uses the ``IsUrban`` and ``Calibrated_pop`` columns of the main GeoDataFrame (:attr:`gdf`) and the
        ``Rural_HHsize`` and ``Urban_HHsize`` values from the :attr:`specs` dictionary.

        See also
        --------
        calibrate_urban_rural_split
        calibrate_urban_manual
        calibrate_current_pop
        read_scenario_data
        """
        self.gdf.loc[self.gdf["IsUrban"] < 20, 'Households'] = self.gdf.loc[
                                                                   self.gdf["IsUrban"] < 20, 'Calibrated_pop'] / \
                                                               self.specs["Rural_HHsize"]
        self.gdf.loc[self.gdf["IsUrban"] > 20, 'Households'] = self.gdf.loc[
                                                                   self.gdf["IsUrban"] > 20, 'Calibrated_pop'] / \
                                                               self.specs["Urban_HHsize"]

    def get_value_of_time(self):
        """
        Calculates the value of time based on the minimum wage ($/h) and a spatial representation of wealth.

        Time is monetized using the minimum wage in the study area, defined in the :attr:`specs` dictionary, and a
        geospatial representation of wealth, which can be either a relative wealth index or a poverty layer (see the
        :meth:`extract_wealth_index` method). The minimum wage value is then distributed spatially using an upper limit
        of 0.5 times the minimung wage in the wealthier regions and an lower limit of 0.2 in the poorest regions.
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

    def run(self, technologies: Union[list[str], str] = 'all', restriction: bool = True):
        """Runs the model using the defined ``technologies`` as options to cook with.

        It loops through the ``technologies`` and calculates all costs, benefit and the net-benefit of cooking with
        such technology relative to the current situation in every grid cell of the study area. Then, it calls the
        :meth:`maximum_net_benefit` method to get the technology with highest net-benefit in each cell (and saves it
        in the ``max_benefit_tech`` column of the :attr:`gdf`). Finally, it extracts indicators such as lives saved,
        time saved, avoided emissions, health costs saved, opportunity cost gained, investment costs, fuel costs, and
        O&M costs.

        Parameters
        ----------
        technologies: str or list of str, default 'all'
            List of technologies to use for the analysis. If 'all' all technologies inside the :attr:`techs` attribute
            would be used. If a list of technology names is passed, then those technologies only will be used.

            .. Note::
               All technology names passed need to match the names of technologies in the :attr:`techs` dictionary.

        restriction: bool, default True
            Whether to have the restriction to only select technologies that produce a positive benefit compared to the
            baseline.

        See also
        --------
        maximum_net_benefit
        extract_lives_saved
        extract_health_costs_saved
        extract_time_saved
        extract_opportunity_cost
        extract_reduced_emissions
        extract_emissions_costs_saved
        extract_investment_costs
        extract_fuel_costs
        extract_om_costs
        extract_salvage
        """
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
            if not tech.is_base:
                tech.adjusted_pm25()
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
            tech.net_benefit(self, self.specs['w_health'], self.specs['w_spillovers'],
                             self.specs['w_environment'], self.specs['w_time'], self.specs['w_costs'])

        print('Getting maximum net benefit technologies...')
        self.maximum_net_benefit(techs, restriction=restriction)
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

    # TODO: check if this function is still needed
    def _get_column_functs(self):
        columns_dict = {column: 'first' for column in self.gdf.columns}
        for column in self.gdf.columns[self.gdf.columns.str.contains('cost|benefit|pop|Pop|Households')]:
            columns_dict[column] = 'sum'
        columns_dict['max_benefit_tech'] = 'first'
        return columns_dict

    def maximum_net_benefit(self, techs: list['Technology'], restriction: bool = True):
        """Extracts the technology or technology combinations producing the highest net-benefit in each cell.

        It saves the technology with highest net-benefit in the ``max_benefi_tech`` column of the :attr:`gdf`
        GeoDataframe.

        Parameters
        ----------
        techs: list of Technology like objects
            Technologies to compare and select the one, or combination of two, that produces the highest net-benefit in
            each cell.
        restriction: bool, default True
            Whether to have the restriction to only select technologies that produce a positive benefit compared to the
            baseline.

        See also
        --------
        run
        """
        net_benefit_cols = [col for col in self.gdf if 'net_benefit_' in col]
        benefits_cols = [col for col in self.gdf if 'benefits_' in col]

        for benefit, net in zip(benefits_cols, net_benefit_cols):
            self.gdf[net + '_temp'] = self.gdf[net]
            if restriction in [True, 'yes', 'y', 'Y', 'Yes', 'PositiveBenefits', 'Positive_Benefits']:
                self.gdf.loc[self.gdf[benefit] < 0, net + '_temp'] = np.nan

        temps = [col for col in self.gdf if '_temp' in col]
        self.gdf["max_benefit_tech"] = self.gdf[temps].idxmax(axis=1).astype('string')

        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("net_benefit_", "")
        self.gdf['max_benefit_tech'] = self.gdf['max_benefit_tech'].str.replace("_temp", "")
        self.gdf["maximum_net_benefit"] = self.gdf[temps].max(axis=1)

        gdf = gpd.GeoDataFrame()
        # gdf = gdf.astype(dtype=gdf.dtypes.to_dict())
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

                second_tech_net_benefit = dff.loc[current, second_benefit_cols].max(axis=1) * (
                        1 - tech.factor.loc[current])

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
                gdf = pd.concat([gdf, dff])

        self.gdf = pd.concat([self.gdf, gdf])

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

    # TODO: check if we ned this method
    def _add_admin_names(self, admin, column_name):
        if isinstance(admin, str):
            admin = gpd.read_file(admin)

        admin.to_crs(self.gdf.crs, inplace=True)

        self.gdf = gpd.sjoin(self.gdf, admin[[column_name, 'geometry']], how="inner", op='intersects')
        self.gdf.drop('index_right', axis=1, inplace=True)
        self.gdf.sort_index(inplace=True)

    def extract_lives_saved(self):
        """Extracts the number of deaths avoided from adopting each stove type selected across the study area and saves
        the data in the ``deaths_avoided`` column of the :attr:`gdf`.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "deaths_avoided"] = self.techs[tech].deaths_avoided[index]

    def extract_health_costs_saved(self):
        """
        Extracts the health costs avoided from adopting each stove type selected across the study area. The health costs
        includes costs of avoided deaths, sickness and spillovers.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "health_costs_avoided"] = self.techs[tech].distributed_morbidity[index] + \
                                                            self.techs[tech].distributed_mortality[index] + \
                                                            self.techs[tech].distributed_spillovers_morb[index] + \
                                                            self.techs[tech].distributed_spillovers_mort[index]

    def extract_time_saved(self):
        """
        Extracts the total time saved from adopting each stove type selected across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "time_saved"] = self.techs[tech].total_time_saved[index]

    def extract_opportunity_cost(self):
        """
        Extracts the opportunity cost of adopting each stove type selected across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "opportunity_cost_gained"] = self.techs[tech].time_value[index]

    def extract_reduced_emissions(self):
        """
        Extracts the reduced emissions achieved by adopting each stove type selected across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "reduced_emissions"] = self.techs[tech].decreased_carbon_emissions[index]

    def extract_investment_costs(self):
        """
        Extracts the total investment costs needed in order to adopt each stove type across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "investment_costs"] = self.techs[tech].discounted_investments[index]

    def extract_om_costs(self):
        """
        Extracts the total operation and maintenance costs needed in order to adopt each stove type across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "om_costs"] = self.techs[tech].discounted_om_costs[index]

    def extract_fuel_costs(self):
        """
        Extracts the total fuel costs needed in order to adopt each stove type across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "fuel_costs"] = self.techs[tech].discounted_fuel_cost[index]

    def extract_salvage(self):
        """
        Extracts the total salvage costs in order to adopt each stove type across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "salvage_value"] = self.techs[tech].discounted_salvage_cost[index]

    def extract_emissions_costs_saved(self):
        """
        Extracts the economic value of the emissions by adopt each stove type across the study area.
        """
        for tech in self.gdf['max_benefit_tech'].unique():
            is_tech = self.gdf['max_benefit_tech'] == tech
            index = self.gdf.loc[is_tech].index
            self.gdf.loc[is_tech, "emissions_costs_saved"] = self.techs[tech].decreased_carbon_costs[index]

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

            gdf.rename(columns={wealth_column: "relative_wealth"}, inplace=True)

            self.gdf = gpd.sjoin(self.gdf, gdf[["relative_wealth", "geometry"]], how="left")
        elif file_type == "raster":
            layer = RasterLayer('Demographics', 'Wealth', path=wealth_index, resample='average')

            layer.align(self.base_layer.path)

            self.raster_to_dataframe(layer, name="relative_wealth", method='read',
                                     fill_nodata_method='interpolate')
        else:
            raise ValueError("file_type needs to be either csv, raster, polygon or point.")

    @staticmethod
    def _re_name(df, labels, variable):
        if labels is not None:
            for value, label in labels.items():
                df[variable] = df[variable].str.replace('_', ' ')
                df.loc[df[variable] == value, variable] = label
            return df

    def _points_to_raster(self, dff, variable, cell_width=1000, cell_height=1000,
                          dtype=rasterio.uint8, nodata=111):
        bounds = self.mask_layer.bounds
        height, width = RasterLayer.shape_from_cell(bounds, cell_height, cell_width)
        transform = rasterio.transform.from_bounds(*bounds, width, height)
        rasterized = features.rasterize(
            ((g, v) for v, g in zip(dff[variable].values, dff['geometry'].values)),
            out_shape=(height, width),
            transform=transform,
            all_touched=True,
            fill=nodata,
            dtype=dtype)
        meta = dict(driver='GTiff',
                    dtype=dtype,
                    count=1,
                    crs=self.mask_layer.data.crs,
                    width=width,
                    height=height,
                    transform=transform,
                    nodata=nodata,
                    compress='DEFLATE')
        return rasterized, meta

    @staticmethod
    def _empty_raster_from_shape(crs, transform, height, width):
        array = np.empty((height, width))
        array[:] = np.nan
        raster = RasterLayer()
        raster.data = array
        raster.meta = dict(crs=crs,
                           dtype=float,
                           width=width,
                           height=height,
                           nodata=np.nan,
                           transform=transform)
        return raster

    def _base_layer_from_bounds(self, bounds, cell_height, cell_width):
        if 'index' not in self.gdf.columns:
            dff = self.gdf.copy().reset_index(drop=False)
        else:
            dff = self.gdf
        height, width = RasterLayer.shape_from_cell(bounds, cell_height, cell_width)
        transform = rasterio.transform.from_bounds(*bounds, width, height)
        geometry = dff["geometry"].apply(lambda geom: geom.wkb)
        gdf = dff.loc[geometry.drop_duplicates().index]
        rows, cols = rasterio.transform.rowcol(transform, gdf['geometry'].x, gdf['geometry'].y)
        self.rows = np.array(rows)
        self.cols = np.array(cols)
        self.base_layer = self._empty_raster_from_shape(self.gdf.crs, transform, height, width)

    def create_layer(self, variable: str, name: Optional[str] = None,
                     labels: Optional[dict[str, str]] = None, cmap: Optional[dict[str, str]] = None,
                     metric: str = 'mean') -> tuple[RasterLayer, dict[int, str], dict[int, str]]:
        """Creates a :class:`RasterLayer` from a column of the main GeoDataFrame (:attr:`gdf`).

        If the data is categorical, then a rasterized version of the data is created, using integers in asscending
        order for the diffenrent unique categories found. A ``codes`` and a ``cmap`` dictionaries will be returned
        containing the names and color equivalent of the numbered categories.

        If the data is non-categorical, the source data will be rasterized into the :class:`RasterLayer` using one of
        the available ``metrics``.

        Parameters
        ----------
        variable: str
            The column name from the :attr:`gdf` to use.
        name: str, optional
            The name to give to the :class:`RasterLayer`.
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for the data categories. It is only used for categorical data.

            .. code-block:: python
               :caption: Example of labels dictionary

               >>> labels = {'Biogas and Electricity': 'Electricity and Biogas',
               ...           'Collected Traditional Biomass': 'Biomass',
               ...           'Collected Improved Biomass': 'Biomass ICS (ND)',
               ...           'Traditional Charcoal': 'Charcoal',
               ...           'Biomass Forced Draft': 'Biomass ICS (FD)',
               ...           'Pellets Forced Draft': 'Pellets ICS (FD)'}

        cmap: dictionary of str key-value pairs, optional
            Dictionary with the colors to use for each data category. It is only used for categorical data.

            .. code-block:: python
               :caption: Example of cmap dictionary

               >>> cmap = {'Biomass ICS (ND)': '#6F4070',
               ...         'LPG': '#66C5CC',
               ...         'Biomass': '#FFB6C1',
               ...         'Biomass ICS (FD)': '#af04b3',
               ...         'Pellets ICS (FD)': '#ef02f5',
               ...         'Charcoal': '#364135',
               ...         'Charcoal ICS': '#d4bdc5',
               ...         'Biogas': '#73AF48',
               ...         'Biogas and Biomass ICS (ND)': '#F6029E',
               ...         'Biogas and Biomass ICS (FD)': '#F6029E',
               ...         'Biogas and Pellets ICS (FD)': '#F6029E',
               ...         'Biogas and LPG': '#0F8554',
               ...         'Biogas and Biomass': '#266AA6',
               ...         'Biogas and Charcoal': '#3B05DF',
               ...         'Biogas and Charcoal ICS': '#3B59DF',
               ...         'Electricity': '#CC503E',
               ...         'Electricity and Biomass ICS (ND)': '#B497E7',
               ...         'Electricity and Biomass ICS (FD)': '#B497E7',
               ...         'Electricity and Pellets ICS (FD)': '#B497E7',
               ...         'Electricity and LPG': '#E17C05',
               ...         'Electricity and Biomass': '#FFC107',
               ...         'Electricity and Charcoal ICS': '#660000',
               ...         'Electricity and Biogas': '#f97b72',
               ...         'Electricity and Charcoal': '#FF0000'}

        metric: str, default 'mean'
            Metric to use to aggregate data. It is only used for non-categorical data. Available metrics:

            * ``mean``: average value between technologies used in the same cell.
            * ``total``: the total value of the data accounting for all households in the cell.
            * ``per_100k``: the values are calculated per 100 thousand population withing each cell.
            * ``per_household``: average value per househol in each cell.

        Returns
        -------
        raster: RasterLayer
            The :class:RasterLayer object.
        codes: dictionary of int-str pairs
            Contains the name equivalent to therasterized data (used if the data is categorical).
        cmap: dictionary of int-str pairs
            A modified cmap containing the color s equivalent to the rasterized data (used if the data is categorical).
        """
        codes = None
        if self.base_layer is not None:
            layer = np.empty(self.base_layer.data.shape)
            layer[:] = np.nan
            dff = self.gdf.copy().reset_index(drop=False)
        else:
            layer = None
            dff = self.gdf.copy()

        if isinstance(self.gdf[variable].iloc[0], str):
            if isinstance(labels, dict):
                dff = self._re_name(dff, labels, variable)
            dff[variable] += ' and '
            dff = dff.groupby('index').agg({variable: 'sum', 'geometry': 'first'})
            dff[variable] = [s[0:len(s) - 5] for s in dff[variable]]
            if isinstance(labels, dict):
                dff = self._re_name(dff, labels, variable)
            codes = {tech: i for i, tech in enumerate(dff[variable].unique())}

            if isinstance(cmap, dict):
                cmap = {i: cmap[tech] for i, tech in enumerate(dff[variable].unique())}

            if self.rows is not None:
                layer[self.rows, self.cols] = [codes[tech] for tech in dff[variable]]
                meta = self.base_layer.meta
            else:
                dff['codes'] = [codes[tech] for tech in dff[variable]]
                layer, meta = self._points_to_raster(dff, 'codes')
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
            else:
                layer, meta = self._points_to_raster(dff, variable, dtype='float32',
                                                     nodata=np.nan)
            variable = variable + '_' + metric
        if name is not None:
            variable = name
        raster = RasterLayer('Output', variable)
        raster.data = layer
        raster.meta = meta

        return raster, codes, cmap

    def to_raster(self, variable: str,
                  labels: Optional[dict[str, str]] = None,
                  cmap: Optional[dict[str, str]] = None,
                  metric: str = 'mean'):
        """Creates a RasterLayer and saves it as a ``.tif`` file and a ``.clr`` colormap.

        Parameters
        ----------
        variable: str
            The column name from the :attr:`gdf` to use.
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for the data categories. It is only used for categorical data---
            see :meth:`create_layer`.
        cmap: dictionary of str key-value pairs, optional
            Dictionary with the colors to use for each data category. It is only used for categorical data---see
            :meth:`create_layer`.
        metric: str, default 'mean'
            Metric to use to aggregate data. It is only used for non-categorical data. For available metrics see
            :meth:`create_layer`.
        """
        raster, codes, cmap = self.create_layer(variable, labels=labels, cmap=cmap, metric=metric)
        raster.save(os.path.join(self.output_directory, 'Output'))
        print(f'Layer saved in {os.path.join(self.output_directory, "Output", variable + ".tif")}\n')
        if codes and cmap:
            with open(os.path.join(self.output_directory, 'Output', f'{variable}ColorMap.clr'), 'w') as f:
                for label, code in codes.items():
                    r = int(to_rgb(cmap[code])[0] * 255)
                    g = int(to_rgb(cmap[code])[1] * 255)
                    b = int(to_rgb(cmap[code])[2] * 255)
                    f.write(f'{code} {r} {g} {b} 255 {label}\n')

    def plot(self, variable: str, metric='mean',
             labels: Optional[dict[str, str]] = None,
             cmap: Union[dict[str, str], str] = 'viridis',
             cumulative_count: Optional[tuple[float, float]] = None,
             quantiles: Optional[tuple[float]] = None,
             admin_layer: Optional[Union[gpd.GeoDataFrame, VectorLayer]] = None,
             title: Optional[str] = None,
             legend: bool = True, legend_title: str = '', legend_cols: int = 1,
             legend_position: tuple[float, float] = (1.05, 1),
             legend_prop: dict = {'title': {'size': 12, 'weight': 'bold'}, 'size': 12},
             stats: bool = False, stats_position: tuple[float, float] = (1.05, 0.5), stats_fontsize: int = 12,
             scale_bar: Optional[dict] = None, north_arrow: Optional[dict] = None,
             ax: Optional['matplotlib.axes.Axes'] = None,
             figsize: tuple[float, float] = (6.4, 4.8),
             rasterized: bool = True,
             dpi: float = 150,
             save_style: bool = False, style_classes: int = 5):
        """Plots a map from a desired column ``variable`` from the :attr:`gdf`.

        Parameters
        ----------
        variable: str
            The column name from the :attr:`gdf` to use.
        metric: str, default 'mean'
            Metric to use to aggregate data. It is only used for non-categorical data. For available metrics see
            :meth:`create_layer`.
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for the data categories. It is only used for categorical data---
            see :meth:`create_layer`.
        cmap: dictionary of str key-value pairs or str, default 'viridis'
            Dictionary with the colors to use for each data category if the data is categorical---see
            :meth:`create_layer`. If the data is continuous, then a name af a color scale accepted y
            :doc:`matplotlib<matplotlib:tutorials/colors/colormaps>` should be passed.
        cumulative_count: array-like of float, optional
            List of lower and upper limits to consider for the cumulative count. If defined the map will be displayed
            with the cumulative count representation of the data.

            .. seealso::
               :meth:`RasterLayer.cumulative_count`

        quantiles: array-like of float, optional
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive. If defined the map
            will be displayed with the quantiles representation of the data.

            .. seealso::
               :meth:`RasterLayer.quantiles`

        admin_layer: gpd.GeoDataFrame or VectorLayer, optional
            The administrative boundaries to plot as background. If no ``admin_layer`` is provided then the
            :attr:``mask_layer`` will be used if available, if not then no boundaries will be ploted.

        title: str, optional
            The title of the plot.
        legend: bool, default False
            Whether to display a legend---only applicable for categorical data.
        legend_title: str, default ''
            Title of the legend.
        legend_cols: int, default 1
            Number of columns to divide the rows of the legend.
        legend_position: array-like of float, default (1.05, 1)
            Position of the upper-left corner of the legend measured in fraction of `x` and `y` axis.
        legend_prop: dict
            Dictionary with the font properties of the legend. It can contain any property accepted by the ``prop``
            parameter from :doc:`matplotlib.pyplot.legend<matplotlib:api/_as_gen/matplotlib.pyplot.legend>`. It
            defaults to ``{'title': {'size': 12, 'weight': 'bold'}, 'size': 12}``.
        stats: bool, default False
            Whether to display the statistics of the analysis in the map.
        stats_position: array-like of float, default (1.05, 1)
            Position of the upper-left corner of the statistics box measured in fraction of `x` and `y` axis.
        stats_fontsize: int, default 12
            The font size of the statistics text.
        scale_bar: dict, optional
            Dictionary with the parameters needed to create a :class:`ScaleBar`. If not defined, no scale bar will be
            displayed.

            .. code-block::
               :caption: Scale bar dictionary example

               dict(size=1000000, style='double', textprops=dict(size=8),
                    linekw=dict(lw=1, color='black'), extent=0.01)

            .. Note::
               See :func:`onstove.scale_bar` for more details

        north_arrow: dict, optional
            Dictionary with the parameters needed to create a north arrow icon in the map. If not defined, the north
            icon wont be displayed.

            .. code-block::
               :caption: North arrow dictionary example

               dict(size=30, location=(0.92, 0.92), linewidth=0.5)

            .. Note::
               See :func:`onstove.north_arrow` for more details

        ax: matplotlib.axes.Axes, optional
            A matplotlib axes instance can be passed in order to overlay layers in the same axes.
        figsize: tuple of floats, default (6.4, 4.8)
            The size of the figure in inches.
        rasterized: bool, default True
            Whether to rasterize the output.It converts vector graphics into a raster image (pixels). It can speed up
            rendering and produce smaller files for large data sets---see more at
            :doc:`matplotlib:gallery/misc/rasterization_demo`.
        dpi: int, default 150
            The resolution of the figure in dots per inch.
        save_style: bool, default False
            Whether to save the style of the plot as a ``.sld`` file---see :meth:`onstove.RasterLayer.save_style`.
        style_classes: int, default 5
            number of classes to include in the ``.sld`` style.

        Examples
        --------
        >>> africa = OnStove('results.pkl')
        ...
        >>> cmap = {'Biomass ICS (ND)': '#6F4070',
        ...         'LPG': '#66C5CC',
        ...         'Biomass': '#FFB6C1',
        ...         'Biomass ICS (FD)': '#af04b3',
        ...         'Pellets ICS (FD)': '#ef02f5',
        ...         'Charcoal': '#364135',
        ...         'Charcoal ICS': '#d4bdc5',
        ...         'Biogas': '#73AF48',
        ...         'Biogas and Biomass ICS (ND)': '#F6029E',
        ...         'Biogas and Biomass ICS (FD)': '#F6029E',
        ...         'Biogas and Pellets ICS (FD)': '#F6029E',
        ...         'Biogas and LPG': '#0F8554',
        ...         'Biogas and Biomass': '#266AA6',
        ...         'Biogas and Charcoal': '#3B05DF',
        ...         'Biogas and Charcoal ICS': '#3B59DF',
        ...         'Electricity': '#CC503E',
        ...         'Electricity and Biomass ICS (ND)': '#B497E7',
        ...         'Electricity and Biomass ICS (FD)': '#B497E7',
        ...         'Electricity and Pellets ICS (FD)': '#B497E7',
        ...         'Electricity and LPG': '#E17C05',
        ...         'Electricity and Biomass': '#FFC107',
        ...         'Electricity and Charcoal ICS': '#660000',
        ...         'Electricity and Biogas': '#f97b72',
        ...         'Electricity and Charcoal': '#FF0000'}
        ...
        >>>   labels = {'Biogas and Electricity': 'Electricity and Biogas',
        ...             'Collected Traditional Biomass': 'Biomass',
        ...             'Collected Improved Biomass': 'Biomass ICS (ND)',
        ...             'Traditional Charcoal': 'Charcoal',
        ...             'Biomass Forced Draft': 'Biomass ICS (FD)',
        ...             'Pellets Forced Draft': 'Pellets ICS (FD)'}
        ...
        >>> scale_bar_prop = dict(size=1000000, style='double', textprops=dict(size=8),
        ...                       linekw=dict(lw=1, color='black'), extent=0.01)
        >>> north_arow_prop = dict(size=30, location=(0.92, 0.92), linewidth=0.5)
        ...
        >>> africa.plot('max_benefit_tech', labels=labels, cmap=cmap,
        ...             stats=True, stats_position=(-0.002, 0.61), stats_fontsize=10,
        ...             legend=True, legend_position=(0.03, 0.47),
        ...             legend_title='Maximum benefit cooking technology',
        ...             legend_prop={'title': {'size': 10, 'weight': 'bold'}, 'size': 10},
        ...             scale_bar=scale_bar_prop, north_arrow=north_arow_prop,
        ...             figsize=(16, 9), dpi=300, rasterized=True)

        .. figure:: ../images/max_benefit_tech.png
           :width: 700
           :alt: max benefit cooking technology over SSA created with OnStove
           :align: center

        See also
        --------
        create_layer
        to_image
        to_raster
        RasterLayer.plot
        VectorLayer.plot
        """
        raster, codes, cmap = self.create_layer(variable, labels=labels, cmap=cmap, metric=metric)
        if isinstance(admin_layer, gpd.GeoDataFrame):
            admin_layer = admin_layer
        elif isinstance(self.mask_layer, VectorLayer):
            admin_layer = self.mask_layer.data
        else:
            admin_layer = None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if stats:
            self._add_statistics(ax, stats_position, stats_fontsize)

        raster.plot(cmap=cmap, cumulative_count=cumulative_count,
                    quantiles=quantiles,
                    categories=codes, legend_position=legend_position,
                    admin_layer=admin_layer, title=title, legend=legend,
                    legend_title=legend_title, legend_cols=legend_cols, rasterized=rasterized,
                    ax=ax, legend_prop=legend_prop, scale_bar=scale_bar, north_arrow=north_arrow)

        if save_style:
            if codes:
                categories = {v: f"{v} = {k}" for k, v in codes.items()}
                quantiles = None
            else:
                categories = False
            raster.save_style(os.path.join(self.output_directory, 'Output'),
                              cmap=cmap, quantiles=quantiles, categories=categories,
                              classes=style_classes)

    def _add_statistics(self, ax, stats_position, fontsize=12):
        summary = self.summary(total=True, pretty=False)
        deaths = TextArea("Deaths avoided", textprops=dict(fontsize=fontsize, color='black'))
        health = TextArea("Health costs avoided", textprops=dict(fontsize=fontsize, color='black'))
        emissions = TextArea("Emissions avoided", textprops=dict(fontsize=fontsize, color='black'))
        time = TextArea("Time saved", textprops=dict(fontsize=fontsize, color='black'))

        texts_vbox = VPacker(children=[deaths, health, emissions, time], pad=0, sep=6)

        deaths_avoided = summary.loc['total', 'deaths_avoided']
        health_costs_avoided = summary.loc['total', 'health_costs_avoided'] / 1000
        reduced_emissions = summary.loc['total', 'reduced_emissions']
        time_saved = summary.loc['total', 'time_saved']

        deaths = TextArea(f"{deaths_avoided:,.0f} pp/yr", textprops=dict(fontsize=fontsize, color='black'))
        health = TextArea(f"{health_costs_avoided:,.2f} BUSD", textprops=dict(fontsize=fontsize, color='black'))
        emissions = TextArea(f"{reduced_emissions:,.2f} Mton", textprops=dict(fontsize=fontsize, color='black'))
        time = TextArea(f"{time_saved:,.2f} h/hh.day", textprops=dict(fontsize=fontsize, color='black'))

        values_vbox = VPacker(children=[deaths, health, emissions, time], pad=0, sep=6, align='right')

        hvox = HPacker(children=[texts_vbox, values_vbox], pad=0, sep=6)

        ab = AnnotationBbox(hvox, stats_position,
                            xycoords='axes fraction',
                            box_alignment=(0, 1),
                            pad=0.0,
                            bboxprops=dict(boxstyle='round',
                                           facecolor='#f1f1f1ff',
                                           edgecolor='lightgray'))

        ax.add_artist(ab)

    def to_image(self, variable, name=None, type='png', cmap='viridis', cumulative_count=None, quantiles=None,
                 legend_position=(1.05, 1), admin_layer=None, title=None, dpi=300, labels=None, legend=True,
                 legend_title='', legend_cols=1, rasterized=True, stats=False, stats_position=(1.05, 0.5),
                 stats_fontsize=12, metric='mean', scale_bar=None, north_arrow=None, figsize=(6.4, 4.8),
                 legend_prop={'title': {'size': 12, 'weight': 'bold'}, 'size': 12}):
        raster, codes, cmap = self.create_layer(variable, name=name, labels=labels, cmap=cmap, metric=metric)
        if isinstance(admin_layer, gpd.GeoDataFrame):
            admin_layer = admin_layer
        elif not admin_layer:
            admin_layer = self.mask_layer.data

        if stats:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            self._add_statistics(ax, stats_position, stats_fontsize)
        else:
            ax = None

        raster.save_image(self.output_directory, type=type, cmap=cmap, cumulative_count=cumulative_count,
                          quantiles=quantiles, categories=codes, legend_position=legend_position,
                          admin_layer=admin_layer, title=title, ax=ax, dpi=dpi,
                          legend=legend, legend_title=legend_title, legend_cols=legend_cols, rasterized=rasterized,
                          scale_bar=scale_bar, north_arrow=north_arrow, legend_prop=legend_prop)


    # TODO: make this a property
    def summary(self, total=True, pretty=True, labels=None):
        dff = self.gdf.copy()
        if labels is not None:
            dff = self._re_name(dff, labels, 'max_benefit_tech')
        for attribute in ['maximum_net_benefit', 'deaths_avoided', 'health_costs_avoided', 'time_saved',
                          'opportunity_cost_gained', 'reduced_emissions', 'emissions_costs_saved',
                          'investment_costs', 'fuel_costs', 'om_costs', 'salvage_value']:
            dff[attribute] *= dff['Households']
        summary = dff.groupby(['max_benefit_tech']).agg({'Calibrated_pop': lambda row: np.nansum(row) / 1000000,
                                                         'Households': lambda row: np.nansum(row) / 1000000,
                                                         'maximum_net_benefit': lambda row: np.nansum(row) / 1000000,
                                                         'deaths_avoided': 'sum',
                                                         'health_costs_avoided': lambda row: np.nansum(row) / 1000000,
                                                         'time_saved': 'sum',
                                                         'opportunity_cost_gained': lambda row: np.nansum(
                                                             row) / 1000000,
                                                         'reduced_emissions': lambda row: np.nansum(row) / 1000000000,
                                                         'emissions_costs_saved': lambda row: np.nansum(row) / 1000000,
                                                         'investment_costs': lambda row: np.nansum(row) / 1000000,
                                                         'fuel_costs': lambda row: np.nansum(row) / 1000000,
                                                         'om_costs': lambda row: np.nansum(row) / 1000000,
                                                         'salvage_value': lambda row: np.nansum(row) / 1000000,
                                                         }).reset_index()
        if total:
            total = summary[summary.columns[1:]].sum().rename('total')
            total['max_benefit_tech'] = 'total'
            summary = pd.concat([summary, total.to_frame().T])

        summary['time_saved'] /= (summary['Households'] * 1000000 * 365)
        if pretty:
            summary.rename(columns={'max_benefit_tech': 'Max benefit technology',
                                    'Calibrated_pop': 'Population (Million)',
                                    'Households': 'Households (Millions)',
                                    'maximum_net_benefit': 'Total net benefit (MUSD)',
                                    'deaths_avoided': 'Total deaths avoided (pp/yr)',
                                    'health_costs_avoided': 'Health costs avoided (MUSD)',
                                    'time_saved': 'hours/hh.day',
                                    'opportunity_cost_gained': 'Opportunity cost avoided (MUSD)',
                                    'reduced_emissions': 'Reduced emissions (Mton CO2eq)',
                                    'emissions_costs_saved': 'Emissions costs saved (MUSD)',
                                    'investment_costs': 'Investment costs (MUSD)',
                                    'fuel_costs': 'Fuel costs (MUSD)',
                                    'om_costs': 'O&M costs (MUSD)',
                                    'salvage_value': 'Salvage value (MUSD)'}, inplace=True)

        return summary

    def plot_split(self, labels: Optional[dict[str, str]] = None,
                   cmap: Optional[dict[str, str]] = None,
                   x_variable: str = 'Calibrated_pop',
                   height: float = 1.5, width: float = 2.5,
                   save_as: Optional[bool] = None):
        """Displays a bar plot with the population or households share using the technologies with highest net-benefits
        over the study area.

        Parameters
        ----------
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for the data categories.

            .. code-block:: python
               :caption: Example of labels dictionary

               >>> labels = {'Collected Traditional Biomass': 'Biomass',
               ...           'Collected Improved Biomass': 'Biomass ICS (ND)',
               ...           'Traditional Charcoal': 'Charcoal',
               ...           'Biomass Forced Draft': 'Biomass ICS (FD)',
               ...           'Pellets Forced Draft': 'Pellets ICS (FD)'}

        cmap: dictionary of str key-value pairs, optional
            Dictionary with the colors to use for each technology.

            .. code-block:: python
               :caption: Example of cmap dictionary

               >>> cmap = {'Biomass ICS (ND)': '#6F4070',
               ...         'LPG': '#66C5CC',
               ...         'Biomass': '#FFB6C1',
               ...         'Biomass ICS (FD)': '#af04b3',
               ...         'Pellets ICS (FD)': '#ef02f5',
               ...         'Charcoal': '#364135',
               ...         'Charcoal ICS': '#d4bdc5',
               ...         'Biogas': '#73AF48'}

        x_variable: str, default 'Calibrated_pop'
            The variable to use in the x axis. Two options are available ``Calibrated_pop`` and ``Households``.
        height: float, default 1.5
            The heihg of the figure in inches.
        width: float, default 2.5
            The width of the figure in inches.
        save_as: str, optional
            If a string is passed, then the plot will be saved with that name as a ``pdf`` file in the
            :attr:`output_directory`.
        """
        df = self.summary(total=False, pretty=False, labels=labels)

        variables = {'Calibrated_pop': 'Population (Millions)', 'Households': 'Households (Millions)'}

        tech_list = df.sort_values(x_variable)['max_benefit_tech'].tolist()
        ccolor = 'black'

        p = (ggplot(df)
             + geom_col(aes(x='max_benefit_tech', y=x_variable, fill='max_benefit_tech'))
             + geom_text(aes(y=df[x_variable], x='max_benefit_tech',
                             label=df[x_variable] / df[x_variable].sum()),
                         format_string='{:.0%}',
                         color=ccolor, size=8, va='center', ha='left')
             + ylim(0, df[x_variable].max() * 1.15)
             + scale_x_discrete(limits=tech_list)
             + scale_fill_manual(cmap)
             + coord_flip()
             + theme_minimal()
             + theme(legend_position='none')
             + labs(x='', y=variables[x_variable], fill='Cooking technology')
             )
        if save_as is not None:
            file = os.path.join(self.output_directory, f'{save_as}.pdf')
            p.save(file, height=height, width=width)
        else:
            return p

    def plot_costs_benefits(self, labels: Optional[dict[str, str]] = None,
                            cmap: Optional[dict[str, str]] = None,
                            height: float = 1.5, width: float = 2.5,
                            save_as: Optional[bool] = None):
        """Displays a stacked bar plot with the aggregated total costs and benefits for the technologies with the
        highest net-benefits over the study area.

        Parameters
        ----------
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for the technology categories.

            .. code-block:: python
               :caption: Example of labels dictionary

               >>> labels = {'Collected Traditional Biomass': 'Biomass',
               ...           'Collected Improved Biomass': 'Biomass ICS (ND)',
               ...           'Traditional Charcoal': 'Charcoal',
               ...           'Biomass Forced Draft': 'Biomass ICS (FD)',
               ...           'Pellets Forced Draft': 'Pellets ICS (FD)'}

        cmap: dictionary of str key-value pairs, optional
            Dictionary with the colors to use for each cost/benefit category.

            .. code-block:: python
               :caption: Example of cmap dictionary

               >>> cmap = {'Health costs avoided': '#542788',
               ...         'Investment costs': '#b35806',
               ...         'Fuel costs': '#f1a340',
               ...         'Emissions costs saved': '#998ec3',
               ...         'Om costs': '#fee0b6',
               ...         'Opportunity cost gained': '#d8daeb'}

        height: float, default 1.5
            The heihg of the figure in inches.
        width: float, default 2.5
            The width of the figure in inches.
        save_as: str, optional
            If a string is passed, then the plot will be saved with that name as a ``pdf`` file in the
            :attr:`output_directory`.
        """
        df = self.summary(total=False, pretty=False, labels=labels)
        df['investment_costs'] -= df['salvage_value']
        df['fuel_costs'] *= -1
        df['investment_costs'] *= -1
        df['om_costs'] *= -1

        value_vars = ['investment_costs', 'fuel_costs', 'om_costs',
                      'health_costs_avoided', 'emissions_costs_saved', 'opportunity_cost_gained']

        dff = df.melt(id_vars=['max_benefit_tech'], value_vars=value_vars)

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

        if save_as is not None:
            file = os.path.join(self.output_directory, f'{save_as}.pdf')
            p.save(file, height=height, width=width)
        else:
            return p

    # TODO: test this method
    # def plot_costs_benefits_unit(self, technologies, area='urban', cmap=None, save=False, height=1.5, width=2.5):
    #     df = pd.DataFrame({'Settlement': [], 'Technology': [], 'investment_costs': [], 'fuel_costs': [],
    #                        'om_costs': [], 'health_costs_avoided': [],
    #                        'emissions_costs_saved': [], 'opportunity_cost_gained': []})
    #
    #     if area.lower() == 'urban':
    #         settlement = self.gdf.reset_index().groupby('index').agg({'IsUrban': 'first'})['IsUrban'] > 20
    #         set_type = 'Urban'
    #     elif area.lower() == 'rural':
    #         settlement = self.gdf.reset_index().groupby('index').agg({'IsUrban': 'first'})['IsUrban'] < 20
    #         set_type = 'Rural'
    #
    #     for name in technologies:
    #         tech = self.techs[name]
    #         total_hh = tech.households[settlement].sum()
    #         inv = np.nansum((tech.discounted_investments[settlement] -
    #                          tech.discounted_salvage_cost[settlement]) * tech.households[settlement]) / total_hh
    #         fuel = np.nansum(tech.discounted_fuel_cost[settlement] * tech.households[settlement]) / total_hh
    #         om = np.nansum(tech.discounted_om_costs[settlement] * tech.households[settlement]) / total_hh
    #         health = np.nansum((tech.distributed_morbidity[settlement] +
    #                             tech.distributed_mortality[settlement] +
    #                             tech.distributed_spillovers_morb[settlement] +
    #                             tech.distributed_spillovers_mort[settlement]) * tech.households[settlement]) / total_hh
    #         ghg = np.nansum(tech.decreased_carbon_costs[settlement] * tech.households[settlement]) / total_hh
    #         time = np.nansum(tech.time_value[settlement] * tech.households[settlement]) / total_hh
    #
    #         df = pd.concat([df, pd.DataFrame({'Settlement': [set_type],
    #                                           'Technology': [name],
    #                                           'Investment costs': [inv],
    #                                           'Fuel costs': [fuel],
    #                                           'O&M costs': [om],
    #                                           'Health costs avoided': [health],
    #                                           'Emissions costs saved': [ghg],
    #                                           'Opportunity cost gained': [time]})])
    #
    #     df['Fuel costs'] *= -1
    #     df['Investment costs'] *= -1
    #     df['O&M costs'] *= -1
    #
    #     if cmap is None:
    #         cmap = {'Health costs avoided': '#542788', 'Investment costs': '#b35806',
    #                 'Fuel costs': '#f1a340', 'Emissions costs saved': '#998ec3',
    #                 'O&M costs': '#fee0b6', 'Opportunity cost gained': '#d8daeb'}
    #
    #     value_vars = ['Health costs avoided',
    #                   'Emissions costs saved',
    #                   'Opportunity cost gained',
    #                   'Investment costs',
    #                   'Fuel costs',
    #                   'O&M costs']
    #
    #     df['net_benefit'] = df['Health costs avoided'] + df['Emissions costs saved'] + df['Opportunity cost gained'] + \
    #                         df['Investment costs'] + df['Fuel costs'] + df['O&M costs']
    #     dff = df.melt(id_vars=['Technology', 'Settlement', 'net_benefit'], value_vars=value_vars)
    #
    #     tech_list = list(dict.fromkeys(dff.sort_values(['Settlement', 'net_benefit'])['Technology'].tolist()))
    #     dff['Technology_cat'] = pd.Categorical(dff['Technology'], categories=tech_list)
    #     cat_order = ['Health costs avoided',
    #                  'Emissions costs saved',
    #                  'Opportunity cost gained',
    #                  'Investment costs',
    #                  'Fuel costs',
    #                  'O&M costs']
    #
    #     dff['variable'] = pd.Categorical(dff['variable'], categories=cat_order, ordered=True)
    #
    #     p = (ggplot(dff)
    #          + geom_col(aes(x='Technology_cat', y='value', fill='variable'))
    #          + geom_point(aes(x='Technology_cat', y='net_benefit'), fill='white', color='grey', shape='D')
    #          + scale_fill_manual(cmap)
    #          + coord_flip()
    #          + theme_minimal()
    #          + labs(x='', y='USD', fill='Cost / Benefit')
    #          + facet_wrap('Settlement',
    #                       nrow=2,
    #                       # scales='free'
    #                       )
    #          )
    #
    #     if save:
    #         file = os.path.join(self.output_directory, 'benefits_costs.pdf')
    #         p.save(file, height=height, width=width)
    #     else:
    #         return p

    def plot_benefit_distribution(self, type: str = 'box', groupby: str = 'None',
                                  variable: str = 'net_benefit', best_mix: bool = True,
                                  labels: Optional[dict[str, str]] = None,
                                  cmap: Optional[dict[str, str]] = None,
                                  height: float = 1.5, width: float = 2.5,
                                  save_as: Optional[bool] = None):
        """Displays a distribution plot with the net-benefits, benefits or costs for the technologies with the
        highest net-benefits throughout the households of the study area.

        Parameters
        ----------
        type: str, default 'box'
            The type of distribution plot to use. Available options are ``box`` and ``density``.
        groupby: str, default 'None'
            Groups the results by urban/rural split. Available options are ``None``, ``isurban`` and ``urban-rural``.
        variable: str, default 'net_benefit'
            Variable to use for the distribution. Available options are ``net_benefit``, ``benefits`` and ``costs``.
        best_mix: bool, default True
            Whether to plot only results for the highest net-benefit technologies, or all technologies.
        labels: dictionary of str key-value pairs, optional
            Dictionary with the keys-value pairs to use for each technology.

            .. code-block:: python
               :caption: Example of labels dictionary

               >>> labels = {'Collected Traditional Biomass': 'Biomass',
               ...           'Collected Improved Biomass': 'Biomass ICS (ND)',
               ...           'Traditional Charcoal': 'Charcoal',
               ...           'Biomass Forced Draft': 'Biomass ICS (FD)',
               ...           'Pellets Forced Draft': 'Pellets ICS (FD)'}

        cmap: dictionary of str key-value pairs, optional
            Dictionary with the colors to use for technology.

            .. code-block:: python
               :caption: Example of cmap dictionary

               >>> cmap = {'Biomass ICS (ND)': '#6F4070',
               ...         'LPG': '#66C5CC',
               ...         'Biomass': '#FFB6C1',
               ...         'Biomass ICS (FD)': '#af04b3',
               ...         'Pellets ICS (FD)': '#ef02f5',
               ...         'Charcoal': '#364135',
               ...         'Charcoal ICS': '#d4bdc5',
               ...         'Biogas': '#73AF48'}

        height: float, default 1.5
            The heihg of the figure in inches.
        width: float, default 2.5
            The width of the figure in inches.
        save_as: str, optional
            If a string is passed, then the plot will be saved with that name as a ``pdf`` file in the
            :attr:`output_directory`.
        """
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
                df = self._re_name(df, labels, 'max_benefit_tech')
                tech_list = df.groupby('max_benefit_tech')[['Calibrated_pop']].sum()
                tech_list = tech_list.reset_index().sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
                x = 'max_benefit_tech'
            elif groupby.lower() == 'urban-rural':
                df = self.gdf.copy()
                df = self._re_name(df, labels, 'max_benefit_tech')
                df['Urban'] = df['IsUrban'] > 20
                df['Urban'].replace({True: 'Urban', False: 'Rural'}, inplace=True)
                x = 'Urban'
            else:
                if best_mix:
                    df = self.gdf.copy()
                    df = self._re_name(df, labels, 'max_benefit_tech')
                    tech_list = df.groupby('max_benefit_tech')[['Calibrated_pop']].sum()
                    tech_list = tech_list.reset_index().sort_values('Calibrated_pop')['max_benefit_tech'].tolist()
                    x = 'max_benefit_tech'
                    if variable == 'net_benefit':
                        # y = '(health_costs_avoided + opportunity_cost_gained + emissions_costs_saved + salvage_value' + \
                        #     ' - investment_costs - fuel_costs - om_costs)'
                        y = 'maximum_net_benefit'
                        title = 'Net benefit per household (kUSD/yr)'
                    elif variable == 'costs':
                        y = 'investment_costs - salvage_value + fuel_costs + om_costs'
                        title = 'Costs per household (kUSD/yr)'
                else:
                    tech_list = []
                    for name, tech in self.techs.items():
                        if tech.benefits is not None:
                            # if 'net_benefit' in tech.__dict__.keys():
                            tech_list.append(name)
                    x = 'tech'
                    if variable == 'net_benefit':
                        y = 'net_benefits'
                        title = 'Net benefit per household (kUSD/yr)'
                    elif variable == 'costs':
                        y = 'costs'
                        title = 'Costs per household (kUSD/yr)'

                    df = pd.DataFrame({x: [], y: []})
                    for tech in tech_list:
                        df = pd.concat([df, pd.DataFrame({x: [tech] * self.techs[tech][y].shape[0],
                                                          y: self.techs[tech][y]})], axis=0)
                    df = self._re_name(df, labels, x)
                    tech_list = df.groupby(x)[[y]].mean()
                    tech_list = tech_list.reset_index().sort_values(y)[x].tolist()

            p = (ggplot(df)
                 + geom_boxplot(aes(x=x,
                                    y=y,
                                    fill=x,
                                    color=x
                                    ),
                                alpha=0.5, outlier_alpha=0.1, raster=True)
                 + scale_fill_manual(cmap)
                 + scale_color_manual(cmap, guide=False)
                 + coord_flip()
                 + theme_minimal()
                 + labs(y=title, fill='Cooking technology')
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
            df = self._re_name(df, labels, 'max_benefit_tech')
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

        if save_as is not None:
            # if groupby.lower() not in ['none', '']:
            #     sufix = f'_{groupby}'
            # else:
            #     sufix = ''
            file = os.path.join(self.output_directory, f'{save_as}.pdf')
            p.save(file, height=height, width=width, dpi=600)
        else:
            return p

    def to_csv(self, name: str):
        """Saves the main GeoDataFrame :attr:`gdf` as a ``.csv`` file into the :attr:`output_directory`.

        Parameters
        ----------
        name: str
            Name of the file.
        """
        name = os.path.join(self.output_directory, name + '.csv')

        pt = self.gdf.copy()

        pt["X"] = pt["geometry"].x
        pt["Y"] = pt["geometry"].y

        df = pd.DataFrame(pt.drop(columns='geometry'))
        df.to_csv(name, index=False)