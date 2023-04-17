"""This module contains the GIS layer classes used in OnStove."""
import os

import numpy as np
import geopandas as gpd
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

import pyproj
import rasterio
from rasterio import warp, features, windows, transform
from matplotlib.colors import ListedColormap, to_rgb, to_hex
from scipy import ndimage
from typing import Optional, Callable, Union
from warnings import warn

from onstove.plotting_utils import scale_bar as scale_bar_func
from onstove.plotting_utils import north_arrow as north_arrow_func
from .raster import *


def try_import():
    try:
        from skimage.graph.mcp import MCP_Geometric
        return MCP_Geometric
    except Exception as e:
        print('Trying import again...')
        time.sleep(1)
        return try_import()


MCP_Geometric = try_import()


class _Layer:
    """Template Layer initializing all common needed attributes.
    """

    def __init__(self, category: Optional[str] = None, name: Optional[str] = '',
                 path: Optional[str] = None, conn: Optional[str] = None,
                 normalization: Optional[str] = 'MinMax', inverse: bool = False,
                 distance_method:  Optional[str] = 'proximity', distance_limit: Optional[float] = None):
        self.category = category
        self.name = name
        self.normalization = normalization
        self.distance_method = distance_method
        self.distance_limit = distance_limit
        self.inverse = inverse
        self.friction = None
        self.distance_raster = None
        self.restrictions = []
        self.weight = 1
        self.path = path
        self.data = None

    def __repr__(self):
        return 'Layer(name=%r)' % self.name

    def __str__(self):
        s = f'Layer\n    - Name: {self.name}\n'
        for attr, value in self.__dict__.items():
            s += f'    - {attr}: {value}\n'
        return s

    def read_layer(self, layer_path, conn=None):
        pass

    @property
    def friction(self) -> 'RasterLayer':
        """:class:`RasterLayer` object containing a friction raster dataset used to compute a travel time map.

        .. seealso::
            :meth:`RasterLayer.travel_time`
        """
        return self._friction

    @friction.setter
    def friction(self, raster):
        if isinstance(raster, str):
            self._friction = RasterLayer(self.category, self.name + ' - friction',
                                         raster)
        elif isinstance(raster, RasterLayer):
            self._friction = raster
        elif raster is None:
            self._friction = None
        else:
            raise ValueError('Raster file type or object not recognized.')


class VectorLayer(_Layer):
    """A ``VectorLayer`` is an object used to read, manipulate and visualize GIS vector data.

    It uses a :doc:`GeoDataFrame<geopandas:docs/reference/geodataframe>` object  to store the georeferenced data in a
    tabular format. It also stores metadata as ``category`` and `name` of the layer, ``normalization`` and ``distance``
    algorithms to use among others. This data structure is used in both the ``MCA`` and the
    :class:`OnStove<onstove.OnStove>` models and the :class:`onstove.DataProcessor` object.

    Parameters
    ----------
    category: str, optional
        Category of the layer. This parameter is useful to group the data into logical categories such as
        "Demographics", "Resources" and "Infrastructure" or "Demand", "Supply" and "Others". This categories are
        particularly relevant for the ``MCA`` analysis.
    name: str, optional
        Name of the dataset. This name will be used as default in the :meth:`save` method as the name of the file.
    path: str, optional
        The relative path to the datafile. This file can be of any type that is accepted by
        :doc:`geopandas:docs/reference/api/geopandas.read_file`.
    conn: sqlalchemy.engine.Connection or sqlalchemy.engine.Engine, optional
        PostgreSQL connection if the layer needs to be read from a database. This accepts any connection type used by
        :doc:`geopandas:docs/reference/api/geopandas.read_postgis`.

        .. seealso::
           :meth:`read_layer` and :meth:`onstove.DataProcessor.set_postgres`

    query: str, optional
        A query string to filter the data. For more information refer to
        :doc:`pandas:reference/api/pandas.DataFrame.query`.

        .. seealso::
           :meth:`read_layer`

    normalization: str, default 'MinMax'
        Sets the default normalization method to use when calling the :meth:`RasterLayer.normalize` for any
        associated :class:`RasterLayer` (for example a distance raster). This is relevant to calculate the
        :attr:`demand_index<onstove.DataProcessor.demand_index>`,
        :attr:`supply_index<onstove.DataProcessor.supply_index>`,
        :attr:`clean_cooking_index<onstove.DataProcessor.clean_cooking_index>` and
        :attr:`assistance_need_index<onstove.DataProcessor.assistance_need_index>` of the ``MCA`` model.
    inverse: str, optional
        Sets the default mode for the normalization algorithm (see :meth:`RasterLayer.normalize`).
    distance_method: str, default 'proximity'
        Sets the default distance algorithm to use when calling the :meth:`get_distance_raster` method.
    distance_limit: Callable object (function or lambda function) with a numpy array as input, optional
        Defines a distance limit or range to consider when calculating the distance raster.

        .. code-block:: python
           :caption: Example: lambda function for distance range between 1,000 and 10,000 meters

           >>> distance_limit = lambda  x: (x >= 1000) & (x <= 10000)

    bbox: tuple, gpd.GeoDataFrame, gpd.GeoSeries or shapely Geometry, optional
        Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry. For more information
        refer to :doc:`geopandas:docs/reference/api/geopandas.read_file`.

    Attributes
    ----------
    friction
    distance_raster
        :class:`RasterLayer` object containing a distance raster dataset calculated by the :meth:`get_distance_raster`
        method using one of the ``distance`` methods.
    restrictions
        List of :class:`RasterLayer` or :class:`VectorLayer` used to restrict areas from the distance calculations.

        .. warning::
            The use of restrictions is under development and it will be available in future releases.

    weight
        Value to weigh the layer's "importance" on the ``MCA`` model. It is initialized with a default value of 1.
    bounds
    style
        Contains a dictionary with the default style parameters for visualizing the layer.

        .. seealso:
            :meth:`plot`
    layer
        :doc:`GeoDataFrame<geopandas:docs/reference/geodataframe>` with the vector data.
    """

    def __init__(self, category: Optional[str] = None,
                 name: Optional[str] = '',
                 path: Optional[str] = None,
                 conn: Optional['sqlalchemy.engine.Connection'] = None,
                 query: Optional[str] = None,
                 normalization: Optional[str] = 'MinMax',
                 inverse: bool = False,
                 distance_method:  Optional[str] = 'proximity',
                 distance_limit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bbox: Optional[gpd.GeoDataFrame] = None):
        """
        Initializes the class with user defined or default parameters.
        """
        self.style = {}
        super().__init__(category=category, name=name,
                         path=path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance_method=distance_method,
                         distance_limit=distance_limit)
        self.read_layer(path=path, conn=conn, bbox=bbox, query=query)

    def __repr__(self):
        return 'Vector' + super().__repr__()

    def __str__(self):
        return 'Vector' + super().__str__()

    @property
    def bounds(self) -> list[float]:
        """Wrapper property to get the  west, south, east, north bounds of the dataset using the ``total_bounds``
        property of ``Pandas``.
        """
        return self.data.total_bounds

    def read_layer(self, path: str,
                   conn: Optional['sqlalchemy.engine.Connection'] = None,
                   bbox: Optional[gpd.GeoDataFrame] = None,
                   query: Optional[str] = None):
        """Reads a dataset from GIS vector data file.

        It works as a wrapper method that will use either the :doc:`geopandas:docs/reference/api/geopandas.read_file`
        function or the :doc:`geopandas:docs/reference/api/geopandas.read_postgis` function to read vector data and
        store the output in the :attr:`data` attribute and the layer path in the ``path`` attribute.

        Parameters
        ----------
        path: str, optional
            The relative path to the datafile. This file can be of any type that is accepted by
            :doc:`geopandas:docs/reference/api/geopandas.read_file`.
        conn: sqlalchemy.engine.Connection or sqlalchemy.engine.Engine, optional
            PostgreSQL connection if the layer needs to be read from a database. This accepts any connection type used by
            :doc:`geopandas:docs/reference/api/geopandas.read_postgis`.
        bbox: tuple, GeoDataFrame, GeoSeries or shapely Geometry, optional
            Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry. For more information
            refer to :doc:`geopandas:docs/reference/api/geopandas.read_file`.
        query: str, optional
            A query string to filter the data. For more information refer to
            :doc:`pandas:reference/api/pandas.DataFrame.query`.
        """
        if path:
            if conn:
                sql = f'SELECT * FROM {path}'
                self.data = gpd.read_postgis(sql, conn)
            else:
                if isinstance(bbox, gpd.GeoDataFrame):
                    bbox = bbox.dissolve()
                elif bbox is not None:
                    raise ValueError('The `bbox` parameter should be of type GeoDataFrame or None, '
                                     f'type {type(bbox)} was given')
                self.data = gpd.read_file(path, bbox=bbox)

            if query:
                self.data = self.data.query(query)
        self.path = path

    def mask(self, mask_layer: gpd.GeoDataFrame, output_path: str = None, keep_geom_type=False):
        """Wrapper for the :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.clip` method.

        Clip points, lines, or polygon geometries to the mask extent.

        Parameters
        ----------
        mask_layer: GeoDataFrame
            A :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame` object.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the clipped dataset is not saved.
        """
        self.data = gpd.clip(self.data, mask_layer.data.to_crs(self.data.crs), keep_geom_type=keep_geom_type)
        if isinstance(output_path, str):
            self.save(output_path)

    def reproject(self, crs: Union[pyproj.CRS, int], output_path: str = None):
        """Wrapper for the :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.to_crs` method.

        Parameters
        ----------
        crs: pyproj.CRS or int
            The value can be anything accepted by :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.to_crs`,
            such as an authority string (eg “EPSG:4326”), a WKT string or an EPSG int.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the clipped dataset is not saved.
        """
        if self.data.crs != crs:
            self.data.to_crs(crs, inplace=True)
        if isinstance(output_path, str):
            self.save(output_path)

    def proximity(self, base_layer: Optional[Union[str, 'RasterLayer']],
                  output_path: Optional[str] = None,
                  create_raster: Optional[bool] = True) -> 'RasterLayer':
        """Calculates a proximity distance raster taking as starting points the vectors of the layer.

        It uses the ``scipy.ndimage.distance_transform_edt`` function to calculate an exact Euclidean distance
        transform.

        Parameters
        ----------
        base_layer: str or RasterLayer
            Raster layer used as a template to calculate the proximity of the vectors to each grid cell in the
            base layer. The ``base_layer`` must be either a str of the path to a raster file or a :class:`RasterLayer`
            object.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the proximity dataset is not saved.
        create_raster: bool, default True
            Boolean condition. If `True`, a :class:`RasterLayer` will be created and stored in the ``distance_raster``
            attribute of the class. If `False`, a :class:`RasterLayer` with the proximity distance calculation is
            returned.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` containing the distance to the nearest point of the current :class:`VectorLayer`.

        See also
        --------
        get_distance_raster
        """
        if isinstance(base_layer, str):
            with rasterio.open(base_layer) as src:
                width = src.width
                height = src.height
                transform = src.transform
        elif isinstance(base_layer, RasterLayer):
            width = base_layer.meta['width']
            height = base_layer.meta['height']
            transform = base_layer.meta['transform']
        else:
            raise ValueError('The `base_layer` (or `raster` if you are using the `get_distance_raster` method) '
                             'must be either a raster file path or a '
                             f'RasterLayer object. {type(base_layer)} was given instead.')

        rasterized = self.rasterize(value=1, width=width, height=height,
                                    transform=transform)

        data = ndimage.distance_transform_edt(1 - rasterized.data.astype(int),
                                              sampling=[rasterized.meta['transform'][0],
                                                        -rasterized.meta['transform'][4]])

        rasterized.meta.update(nodata=np.nan, dtype='float32')

        distance_raster = RasterLayer(self.category,
                                      self.name + '_dist',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)
        distance_raster.data = data
        distance_raster.meta = rasterized.meta

        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster

    def travel_time(self, friction: Optional['RasterLayer'] = None,
                    output_path: Optional[str] = None,
                    create_raster: Optional[bool] = True) -> 'RasterLayer':
        """Creates a travel time map to the nearest polygon in the layer using a friction surface.

        It calculates the minimum time needed to travel to the nearest point defined by the current :class:`VectorLayer`
        using a surface friction :class:`RasterLayer`. The friction dataset, describes how much time, in minutes, is
        needed to travel one meter across each cell over the region.

        Parameters
        ----------
        friction: RasterLayer, optional
            Surface friction layer in minutes per minute traveled across each cell over a region.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the travel time dataset is not saved.
        create_raster: bool, default True
            Boolean condition. If `True`, a :class:`RasterLayer` will be created and stored in the ``distance_raster``
            attribute of the class. If `False`, a :class:`RasterLayer` with the travel time calculation is returned.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the travel time to the nearest polygon in the current :class:`VectorLayer`.

        Notes
        -----
        For more information on surface friction layers see the
        `Malarian Atlas Project <https://malariaatlas.org/explorer>`_.
        """
        if not isinstance(friction, RasterLayer):
            if not isinstance(self.friction, RasterLayer):
                raise ValueError('A friction `RasterLayer` is needed to calculate the travel time distance raster. '
                                 'Please provide a friction dataset of type `RasterLayer` to the `raster` parameter'
                                 'or set it as default in the `friction` attribute of the class '
                                 '(see the `VectorLayer` documentation).')
            else:
                friction = self.friction

        rows, cols = self.start_points(raster=friction)
        distance_raster = friction.travel_time(rows=rows, cols=cols,
                                             output_path=None, create_raster=False)
        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster

    def get_distance_raster(self, method: str = None,
                            raster: Optional[Union[str, 'RasterLayer']] = None,
                            output_path: Optional[str] = None):
        """This method calls the specified distance calculation method.

        It takes a ``method`` as input and calls the right method using either user predefined or default parameters.
        The calculated distance raster is then stored in the :attr:`distance_raster` attribute of the class.

        Parameters
        ----------
        method: str, optional
            name of the method to use for the distance calculation. It can take "proximity", "travel_time" or None as
            options. If "proximity" is used, then the :meth:`proximity` method is called and a :class:`RasterLayer`
            needs to be passed to the ``raster`` parameter as base layer to use for the calculation. If "travel_time"
            is used, then the :meth:`travel_time` method is called and a friction layer can be passed to the ``raster``
            attribute. If None is used, then the predefined ``distance_method`` attribute of the :class:`VectorLayer`
            class is used instead.
        raster: RasterLayer
            Raster layer used as base or friction layer depending on the ``method`` used (see above).
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the distance raster dataset is not
            saved.
        """
        if method is None:
            if self.distance_method is None:
                raise ValueError('Please pass a distance `method` ("proximity" or "travel time") or define the default '
                                 'method in the `distance` attribute of the class.')
            method = self.distance_method

        if method == 'proximity':
            self.proximity(base_layer=raster, output_path=output_path, create_raster=True)

        elif method == 'travel_time':
            self.travel_time(friction=raster, output_path=output_path, create_raster=True)

    def rasterize(self, raster: 'RasterLayer' = None, 
                  attribute: str = None,
                  value: Union[int, float] = 1,
                  width: int = None, height: int = None,
                  transform: 'AffineTransform' = None,
                  cell_width: Union[int, float] = None,
                  cell_height: Union[int, float] = None,
                  nodata: Union[int, float] = 0,
                  all_touched: bool = True,
                  output_path: Optional[str] = None) -> 'RasterLayer':
        """Converts the vector data into a gridded raster dataset.

        It rasterizes the vector data by taking either a transform, the width and height of the image, or the cell
        width and height of the output raster. It uses the
        :doc:`rasterio.features.rasterize<rasterio:api/rasterio.features>`
        function.

        Parameters
        ----------
        attribute: str, optional
            Name of the column in the GeoDataFrame to rasterize. If defined then the values of such column will be
            burned in the output raster.
        value: int or float, default 1
            If ``attribute`` is not defined, then a fixed value to burn can be defined here.
        width: int, optional
            The width of the output raster. This parameter needs to be passed along with the ``height``.
        height: int, optional
            The height of the output raster. This parameter needs to be passed along with the ``width``.
        transform: Affine or sequence of GroundControlPoint or RPC
            Transform suitable for input to AffineTransformer, GCPTransformer, or RPCTransformer according to
            :doc:`rasterio Transform<rasterio:topics/transforms>`.
        cell_width: int of float, optional
            The width of the cell in the output raster. This parameter needs to be passed along with the
            ``cell_height`` parameter.
        cell_height: int of float, optional
            The height of the cell in hte output raster. This parameter needs to be passed along with the
            ``cell_width`` parameter.
        nodata: int of float, default 0
            No data value to be used for the cells with values outside the target values.
        all_touched: bool, default True
            Defines if all cells touched are considered for the rasterization.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the rasterized dataset is not
            saved.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the rasterized dataset of the current :class:`VectorLayer`.
        """
        if isinstance(raster, RasterLayer):
            transform = raster.meta['transform']
            width = raster.meta['width']
            height = raster.meta['height']
        if transform is None:
            if (width is None) or (height is None):
                height, width = RasterLayer.shape_from_cell(self.bounds, cell_height, cell_width)
            transform = rasterio.transform.from_bounds(*self.bounds, width, height)
        elif width is None or height is None:
            height, width = RasterLayer.shape_from_cell(self.bounds, transform[0], -transform[4])

        if attribute:
            shapes = ((g, v) for v, g in zip(self.data[attribute].values, self.data['geometry'].values))
            dtype = type(self.data[attribute].values[0])
        else:
            shapes = ((g, value) for g in self.data['geometry'].values)
            dtype = type(value)

        rasterized = features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            all_touched=all_touched,
            dtype=dtype)
        meta = dict(driver='GTiff',
                    dtype=dtype,
                    count=1,
                    crs=self.data.crs,
                    width=width,
                    height=height,
                    transform=transform,
                    nodata=nodata)
        raster = RasterLayer(name=self.name)
        raster.data = rasterized
        raster.meta = meta

        if output_path:
            raster.save(output_path)

        return raster

    def start_points(self, raster: "RasterLayer") -> tuple[list[int], list[int]]:
        """Gets the indexes of the overlapping cells of the :class:`VectorLayer` with the input :class:`RasterLayer`.

        Uses the :doc:`rasterio.transform.rowcol<rasterio:api/rasterio.transform>` function to get the rows and columns.

        Parameters
        ----------
        raster: RasterLayer
            Raster layer to overlap with the current :class:`VectorLayer`.

        Returns
        -------
        Tuple of lists rows and columns
            Returns a tupe containing two list, the first one with the row indexes and the second one with the column
            indexes.
        """
        row_list = []
        col_list = []
        for index, row in self.data.iterrows():
            rows, cols = rasterio.transform.rowcol(raster.meta['transform'], row["geometry"].x, row["geometry"].y)
            row_list.append(rows)
            col_list.append(cols)

        return row_list, col_list

    def save(self, output_path: str, name: str = None):
        """Saves the current :class:`VectorLayer` into disk.

        It saves the layer in the ``output_path`` defined using the ``name`` attribute as filename and `geojson` as
        extension.

        Parameters
        ----------
        output_path: str
            Output folder where to save the layer.
        name: str, optional
            Name of the file, if not defined then the :attr:`name` attribute is used.
        """
        for column in self.data.columns:
            if isinstance(self.data[column].iloc[0], datetime.date):
                self.data[column] = self.data[column].astype('datetime64')
        if not isinstance(name, str):
            name = self.name
        output_file = os.path.join(output_path,
                                   name + '.geojson')
        os.makedirs(output_path, exist_ok=True)
        self.data.to_file(output_file, driver='GeoJSON')
        self.path = output_file

    def _add_restricted_areas(self, layer_path, layer_type, **kwargs):
        """Adds restricted areas for the calculations."""
        if layer_type == 'vector':
            i = len(self.restrictions) + 1
            self.restrictions.append(VectorLayer(self.category,
                                                 self.name + f' - restriction{i}',
                                                 layer_path, **kwargs))
        elif layer_type == 'raster':
            i = len(self.restrictions) + 1
            self.restrictions.append(RasterLayer(self.category,
                                                 self.name + f' - restriction{i}',
                                                 layer_path, **kwargs))

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None, column=None, style: dict = None,
             legend_kwds=None):
        """Plots a map of the layer using custom styles.

        This is, in principle, a wrapper function for the :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.plot`
        method.

        Parameters
        ----------
        ax: matplotlib axes instance
            A matplotlib axes instance can be passed in order to overlay layers in the same axes.
        style: dict, optional
            Dictionary containing all custom styles wanted for the map. This dictionary can contain any input that is
            accepted by :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.plot`. If not defined, then the
            :attr:`style` attribute is used.
        """
        if legend_kwds is None:
            legend_kwds = {'loc': 'upper left', 'bbox_to_anchor': (1,1)}
        if style is None:
            style = self.style

        if ax is None:
            ax = self.data.plot(label=self.name, column=column, legend=True,
                                legend_kwds=legend_kwds, **style)
            if column is None:
                if 'facecolor' in style.keys():
                    artist = mpatches.Patch(**style, 
                                            label=self.name)
                    lgnd = ax.legend(handles=[artist], **legend_kwds)
                else:
                    lgnd = ax.legend(**legend_kwds)
            else:
                lgnd = ax.get_legend()
            ax.add_artist(lgnd)
        else:
            ax = self.data.plot(ax=ax, label=self.name, legend_kwds=legend_kwds,
                                column=column, legend=True, **style)

            if column is None:
                lgnd = ax.legend(**legend_kwds)
            else:
                lgnd = ax.get_legend()
            ax.add_artist(lgnd)
        return ax

    @staticmethod
    def remove_duplicates(list1, list2):
        for i in list1:
            if i in list2:
                del i
        return list1 + list2


class RasterLayer(_Layer):
    """A ``RasterLayer`` is an object used to read, manipulate and visualize GIS raster data.

    It uses :doc:`rasterio<rasterio:index>` to read GIS raster data, and stores the output in a
    :class:`numpy.ndarray<numpy:reference/arrays.ndarray>` under the ``layer`` attribute, and the metadata of the layer
    in the ``meta`` attribute. It also stores additional metadata as ``category`` and ``name`` of the layer,
    ``normalization`` and ``distance`` algorithms to use among others. This data structure is used in both the ``MCA``
    and the :class:`OnStove<onstove.OnStove>` models and the :class:`onstove.DataProcessor` object.

    Parameters
    ----------
    category: str, optional
        Category of the layer. This parameter is useful to group the data into logical categories such as
        "Demographics", "Resources" and "Infrastructure" or "Demand", "Supply" and "Others". This categories are
        particularly relevant for the ``MCA`` analysis.
    name: str, optional
        Name of the dataset. This name will be used as default in the :meth:`save` method as the name of the file.
    path: str, optional
        The relative path to the datafile. This file can be of any type that is accepted by
        :doc:`rasterio.open()<rasterio:quickstart>`.
    conn: sqlalchemy.engine.Connection or sqlalchemy.engine.Engine, optional
        PostgreSQL connection if the layer needs to be read from a database.

        .. seealso::
           :meth:`read_layer` and :meth:`onstove.DataProcessor.set_postgres`

        .. warning::
           The PostgreSQL database connection is under development for the :class:`RasterLayer` class and it will be
           available in future releases.

    normalization: str, default 'MinMax'
        Sets the default normalization method to use when calling the :meth:`RasterLayer.normalize`. This is relevant
        to calculate the
        :attr:`demand_index<onstove.MCA.demand_index>`,
        :attr:`supply_index<onstove.MCA.supply_index>`,
        :attr:`clean_cooking_index<onstove.MCA.clean_cooking_index>` and
        :attr:`assistance_need_index<onstove.MCA.assistance_need_index>` of the ``MCA`` model.
    inverse: bool, default False
        Sets the default mode for the normalization algorithm (see :meth:`RasterLayer.normalize`).
    distance_method: str, default 'proximity'
        Sets the default distance algorithm to use when calling the :meth:`get_distance_raster` method.
    distance_limit: Callable object (function or lambda function) with a numpy array as input, optional
        Defines a distance limit or range to consider when calculating the distance raster.

        .. code-block:: python
           :caption: Example: lambda function for distance range between 1,000 and 10,000 meters

           >>> distance_limit = lambda  x: (x >= 1000) & (x <= 10000)

    resample: str, default 'nearest'
        Sets the default method to use when resampling the dataset. Resampling occurs when changing the grid cell size
        of the raster, thus the values of the cells need to be readjusted to reflect the new cell size. Several
        sampling methods can be used, and which one to use is dependent on the nature of the data. For a list of the
        accepted methods refer to :doc:`rasterio.enums.Resampling<rasterio:api/rasterio.enums>`.

        .. seealso::
           :meth:`reproject`

    window: instance of rasterio.windows.Window
        A :doc:`Window<rasterio:api/rasterio.windows>` is a view from a rectangular subset of a raster. It is used to
        perform :doc:`windowed reading<rasterio:topics/windowed-rw>` of raster layers. This is useful when working with
        large raster files in order to reduce memory RAM needs or to read only an area of interest from a broader
        raster layer.

        .. seealso::
           :meth:`read_layer`

    rescale: bool, default False
        Sets the default value for the ``rescale`` attribute. This attribute is used in the :meth:`align` method to
        rescale the values of a cell proportionally to the change in size of the cell. This is useful when aligning
        rasters that have different cell sizes and their values can be scaled proportionally.

    Attributes
    ----------
    friction
    distance_raster
        :class:`RasterLayer` object containing a distance raster dataset calculated by the :meth:`get_distance_raster`
        method using one of the ``distance`` methods.
    normalized
        :class:`RasterLayer` object containing a normalized raster dataset calculated by the :meth:`normalize`
        method using one of the ``normalization`` methods.
    restrictions
        List of :class:`RasterLayer` or :class:`VectorLayer` used to restrict areas from the distance calculations.

        .. warning::
           The use of restrictions is under development and it will be available in future releases.

    weight
        Value to weigh the layer's "importance" on the ``MCA`` model. It is initialized with a default value of 1.
    bounds
    style
        Contains the default style parameters for visualizing the layer.

        .. seealso:
           :meth:`plot`

    data
        :class:`numpy.ndarray<numpy:reference/arrays.ndarray> containing the data of the raster layer.
    """

    def __init__(self, category: Optional[str] = None,
                 name: Optional[str] = '',
                 path: Optional[str] = None,
                 conn: Optional['sqlalchemy.engine.Connection'] = None,
                 normalization: Optional[str] = 'MinMax', inverse: bool = False,
                 distance_method:  Optional[str] = 'proximity',
                 distance_limit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 resample: str = 'nearest', window: Optional[windows.Window] = None,
                 rescale: bool = False):
        """
        Initializes the class with user defined or default parameters.
        """
        self.resample = resample
        self.rescale = rescale
        self.meta = {}
        self.normalized = None
        super().__init__(category=category, name=name,
                         path=path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance_method=distance_method,
                         distance_limit=distance_limit)
        self.read_layer(path, conn, window=window)

    def __repr__(self):
        return 'Raster' + super().__repr__()

    def __str__(self):
        return 'Raster' + super().__str__()

    @property
    def bounds(self) -> list[float]:
        """Wrapper property to get the  west, south, east, north bounds of the dataset using the
        :doc:`rasterio.transform.array_bounds<rasterio:api/rasterio.transform>` function of ``rasterio``.
        """
        return transform.array_bounds(self.meta['height'],
                                      self.meta['width'],
                                      self.meta['transform'])

    def read_layer(self, path, conn=None, window=None):
        """Reads a dataset from a GIS raster data file.

        It works as a wrapper method that uses :doc:`rasterio.open()<rasterio:index>` to read raster data and
        store the output in the :attr:`data` attribute, the metadata in the :attr:`meta` attribute and the layer path
        in the ``path`` attribute.

        Parameters
        ----------
        path: str, optional
            The relative path to the datafile. This file can be of any type that is accepted by
            :doc:`geopandas:docs/reference/api/geopandas.read_file`.
        conn: sqlalchemy.engine.Connection or sqlalchemy.engine.Engine, optional
            PostgreSQL connection if the layer needs to be read from a database. This accepts any connection type used by
            :doc:`geopandas:docs/reference/api/geopandas.read_postgis`.
        window: instance of rasterio.windows.Window
            A :doc:`Window<rasterio:api/rasterio.windows>` is a view from a rectangular subset of a raster. It is used to
            perform :doc:`windowed reading<rasterio:topics/windowed-rw>` of raster layers. This is useful when working with
            large raster files in order to reduce memory RAM needs or to read only an area of interest from a broader
            raster layer.
        """
        if path:
            with rasterio.open(path) as src:
                if window is not None:
                    transform = src.transform
                    self.meta = src.meta.copy()
                    window = windows.from_bounds(*window, transform=transform)
                    window_transform = src.window_transform(window)
                    self.data = src.read(1, window=window)
                    self.meta.update(transform=window_transform,
                                     width=self.data.shape[1], height=self.data.shape[0],
                                     compress='DEFLATE')
                else:
                    self.data = src.read(1)
                    self.meta = src.meta

            if self.meta['nodata'] is None:
                if self.data.dtype in [int, 'int32']:
                    self.meta['nodata'] = 0
                    warn(f"The {self.name} layer do not have a defined nodata value, thus 0 was assigned. "
                         f"You can change this defining the nodata value in the metadata of the variable as: "
                         f"variable.meta['nodata'] = value")
                elif self.data.dtype in [float, 'float32', 'float64']:
                    self.meta['nodata'] = np.nan
                    warn(f"The {self.name} layer do not have a defined nodata value, thus np.nan was assigned."
                         f" You can change this defining the nodata value in the metadata of the variable as: "
                         f"variable.meta['nodata'] = value")
                else:
                    warn(f"The {self.name} layer do not have a defined nodata value, please define the nodata "
                         f"value in the metadata of the variable with: variable.meta['nodata'] = value")
        self.path = path

    @staticmethod
    def shape_from_cell(bounds: list[float], cell_height: float, cell_width: float):
        """Gets the shape (width and height) of the raster layer based on the bounds and a cell size.

        Parameters
        ----------
        bounds: list of float
            The west, south, east, north bounds of the raster.

            .. seealso::
                bounds

        cell_height: float
            The cell height in units consistent with the raster's crs.
        cell_width: float
            The cell width in units consistent with the raster's crs.
        """
        height = int((bounds[3] - bounds[1]) / cell_height)
        width = int((bounds[2] - bounds[0]) / cell_width)
        return height, width

    def mask(self, mask_layer: VectorLayer, output_path: Optional[str] = None,
             crop: bool = True, all_touched: bool = False):
        """Creates a masked version of the layer, based on an input shape given by a :class:`VectorLayer`.

        It uses the :doc:`rasterio.mas.mask<rasterio:api/rasterio.mask>` function to create a masked version of the
        raster, based on input shapes that are defined by a :class:`VectorLayer`.

        Parameters
        ----------
        mask_layer: VectorLayer
            Layer that contains the shapes that are used to create the mask. Every pixel of the raster outside the mask,
            will be set to the raster's ``nodata`` value.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the masked dataset is not saved to disk.
        crop: bool, default True
            Boolean indicating if the raster extent should be cropped to the ``mask_layer`` extent.
        all_touched: bool, default True
             Include a pixel in the mask if it touches any of the shapes. If False, include a pixel only if its center
             is within one of the shapes, or if it is selected by Bresenham’s line algorithm.
        """
        rasterized_mask = mask_layer.rasterize(value=1, transform=self.meta['transform'],
                                               width=self.meta['width'], height=self.meta['height'],
                                               nodata=0, all_touched=all_touched)

        self.data[rasterized_mask.data == 0] = self.meta['nodata']

        if crop:
            total_bounds = mask_layer.data['geometry'].total_bounds
            window = windows.from_bounds(*total_bounds, transform=self.meta['transform'])
            height, width = self.shape_from_cell(total_bounds, self.meta['transform'][0], -self.meta['transform'][4])
            row_off = max(int(window.row_off), 0)
            col_off = max(int(window.col_off), 0)
            window = windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=width,
                height=height,
            )
            bounds = rasterio.windows.bounds(window, self.meta['transform'])
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            self.data = self.data[row_off:(row_off + height),
                                  col_off:(col_off + width)]
            self.meta.update(transform=transform, height=height, width=width)

        if output_path:
            self.save(output_path)

        # output_file = os.path.join(output_path,
        #                            self.name + '.tif')
        # mask_raster(self.path, mask_layer.to_crs(self.meta['crs']),
        #             output_file, self.meta['nodata'], 'DEFLATE', all_touched=all_touched)
        # self.read_layer(output_file)

    def reproject(self, crs: rasterio.crs.CRS, output_path: Optional[str] = None,
                  cell_width: Optional[float] = None, cell_height: Optional[float] = None):
        """Reprojects the raster data into a specified coordinate system.

        Uses the :doc:`rasterio.features.rasterize<rasterio:api/rasterio.features>` function to reproject the current
        raster data into a different ``crs``.

        Parameters
        ----------
        crs: rasterio.crs.CRS or dict
            Target coordinate reference system. this can be anything accepted by
            :doc:`rasterio.warp.reproject<rasterio:api/rasterio.warp>`.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the reprojected dataset is not saved
            to disk.
        cell_width: float, optional
            The cell width in units consistent with the raster's crs. If provided the transform of the raster will be
            adjusted accordingly.
        cell_height: float, optional
            The cell height in units consistent with the raster's crs. If provided the transform of the raster will be
            adjusted accordingly.
        """
        if (self.meta['crs'] != crs) or cell_width:
            data, meta = reproject_raster(self.path, crs,
                                          cell_width=cell_width, cell_height=cell_height,
                                          method=self.resample, compression='DEFLATE')
            self.data = data
            self.meta = meta
            if output_path:
                self.save(output_path)

    def calculate_default_transform(self, dst_crs: rasterio.crs.CRS) -> tuple['AffineTransform', int, int]:
        """Wrapper function to calculate the default transform using the
        :doc:`rasterio.warp.calculate_default_transform<rasterio:api/rasterio.warp>` function.

        Parameters
        ----------
        dst_crs: rasterio.crs.CRS or dict
            Target coordinate reference system. this can be anything accepted by
            :doc:`rasterio.warp.reproject<rasterio:api/rasterio.warp>`.

        Returns
        -------
        Tuple of Affine transform, int and int
            Output affine transformation matrix, width and height.
        """
        t, w, h = warp.calculate_default_transform(self.meta['crs'],
                                                   dst_crs,
                                                   self.meta['width'],
                                                   self.meta['height'],
                                                   *self.bounds,
                                                   dst_width=self.meta['width'],
                                                   dst_height=self.meta['height'])
        return t, w, h

    def travel_time(self, rows: np.ndarray, cols: np.ndarray,
                    include_starting_cells: bool = False,
                    output_path: Optional[str] = None,
                    create_raster: Optional[bool] = True) -> 'RasterLayer':
        """Calculates a travel time map using the raster data as cost surface and specific cells as starting points.

        This method uses the data of the current :class:`RasterLayer` as a cost surface, to calculate the
        distance-weighted minimum cost map from specific cells (``rows`` and ``cols``) to every other cell in the cost
        surface. It makes use of the :doc:`skimage.graph.mcp.MCP_Geometric<skimage:api/skimage.graph>` class.

        Parameters
        ----------
        rows: np.ndarray
            Row indexes of the cells to consider as starting points.
        cols: np.ndarray
            Column indexes of the cells to consider as starting points.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the travel time dataset is not saved
            to disk.
        create_raster: bool, default True
            Boolean condition. If `True`, a :class:`RasterLayer` will be created and stored in the ``distance_raster``
            attribute of the class. If `False`, a :class:`RasterLayer` with the travel time calculation is returned.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the least-cost travel time data.
        """
        layer = self.data.copy()
        layer *= (1000 / 60)  # to convert to hours per kilometer
        layer[np.isnan(layer)] = float('inf')
        layer[layer == self.meta['nodata']] = float('inf')
        layer[layer < 0] = float('inf')
        mcp = MCP_Geometric(layer, fully_connected=True)
        pointlist = np.column_stack((rows, cols))
        # TODO: create method for restricted areas
        if len(pointlist) > 0:
            cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
            if include_starting_cells:
                cumulative_costs += layer
            cumulative_costs[np.where(cumulative_costs == float('inf'))] = np.nan
        else:
            cumulative_costs = np.full(self.data.shape, 7.0)

        distance_raster = RasterLayer(self.category,
                                      'traveltime',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)

        meta = self.meta.copy()
        meta.update(nodata=np.nan)
        distance_raster.data = cumulative_costs  # + (self.friction.layer * 1000 / 60)
        distance_raster.meta = meta

        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster

    def log(self, mask_layer: VectorLayer,
            output_path: Optional[str] = None,
            create_raster: Optional[bool] = True) -> 'RasterLayer':
        """Calculates a logarithmic representation of the raster dataset.

        This is used as ``distance`_raster`` for layers that can vary widely in magnitude from cell to cell, like
        `population`. This useful when using the ``MCA`` model.

        Parameters
        ----------
        mask_layer: VectorLayer
            Layer used to set to ``nodata`` every pixel of the raster outside the mask.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the masked dataset is not saved to disk.
        create_raster: bool, default True
            Boolean condition. If `True`, a :class:`RasterLayer` will be created and stored in the ``distance_raster``
            attribute of the class. If `False`, a :class:`RasterLayer` with the logarithmic data is returned.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the logarithmic raster data.
        """
        layer = self.data.copy()
        layer[layer == 0] = np.nan
        layer[layer > 0] = np.log(layer[layer > 0])
        layer = np.nan_to_num(layer, nan=0)

        meta = self.meta.copy()
        meta.update(nodata=np.nan, dtype='float64')

        distance_raster = RasterLayer(self.category,
                                      self.name + ' - log',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)
        distance_raster.data = layer
        distance_raster.meta = meta
        distance_raster.mask(mask_layer)

        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster
            
    def proximity(self, value=1):
        data = self.data.copy()
        data[self.data==value] = 0
        data[self.data!=value] = 1
        data = ndimage.distance_transform_edt(data,
                                              sampling=[self.meta['transform'][0],
                                                        -self.meta['transform'][4]])

        distance_raster = RasterLayer(category=self.category,
                                      name=self.name + '_dist',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)
        distance_raster.data = data
        distance_raster.meta = self.meta
        distance_raster.meta.update(dtype=int)
        return distance_raster

    def get_distance_raster(self, method: Optional[str] = None,
                            output_path: Optional[str] = None,
                            mask_layer: Optional[VectorLayer] = None,
                            starting_points: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """This method calls the specified distance calculation method.

        It takes a ``method`` as input and calls the right method using either user predefined or default parameters.
        The calculated distance raster is then stored in the :attr:`distance_raster` attribute of the class.

        Parameters
        ----------
        method: str, optional
            name of the method to use for the distance calculation. It can take "log", "travel_time" or None as
            options. If "log" is used, then the :meth:`log` method is called and a :class:`VectorLayer`
            needs to be passed to the ``mask_layer`` parameter to be used for the calculation. If "travel_time"
            is used, then the :meth:`travel_time` method is called and the ``starting_points`` `need to be passed.
            If None is used, then the predefined ``distance_method`` attribute of the :class:`RasterLayer` class
            is used instead. If the previous is also None, then the raster it self is used as a distance raster.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the distance raster dataset is not
            saved.
        mask_layer: VectorLayer
            Layer used to set to ``nodata`` every pixel of the raster outside the mask (used with the ``log`` method
            only).
        starting_points: Callable object (function or lambda function) with a numpy array as input, optional
            Function or lambda function describing which are the starting points to consider when calculating the
            travel time map (used with the ``travel_time`` method only).
        """
        if method is None:
            if self.distance_method is None:
                raise ValueError('Please pass a distance `method` ("log" or "travel time") or define the default '
                                 'method in the `distance` attribute of the class.')
            method = self.distance_method

        if method == 'log':
            self.distance_raster = self.log(mask_layer=mask_layer, output_path=output_path)
        elif method == 'travel_time':
            rows, cols = self.start_points(condition=starting_points)
            self.distance_raster = self.travel_time(rows, cols, output_path=output_path)
        else:
            self.distance_raster = self

    def start_points(self, condition: Optional[Callable[[np.ndarray], np.ndarray]]):
        """Gets the rows and columns of the cells tha fulfil a condition.

        Parameters
        ----------
        condition: Callable object with a numpy array as input, or a list of values, optional
            This condition is used to find the cells in the array that are equal to the values

        Returns
        -------
        Tuple of numpy ndarrays
            Tuple with rows and columns arrays containing locations of the cells.
        """
        if callable(condition):
            return np.where(condition(self.data))
        else:
            raise TypeError('The condition can only be a callable object.')

    def normalize(self, output_path: Optional[str] = None,
                  buffer: bool = False, inverse: bool = False,
                  create_raster: Optional[bool] = True) -> 'RasterLayer':
        """Normalizes the raster data by a given normalization method.

        Parameters
        ----------
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the normalized dataset is not saved
            to disk.
        buffer: bool, default False
            Whether to exclude the areas outside the ``distance_limit`` attribute and make them `np.nan`.
        inverse: bool, default False
            Whether to invert the normalized layer or not.
        create_raster: bool, default True
            Boolean condition. If `True`, a :class:`RasterLayer` will be created and stored in the ``normalized``
            attribute of the class. If `False`, a :class:`RasterLayer` with the normalized data is returned.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the normalized data.
        """
        if self.normalization == 'MinMax':
            raster = self.data.copy()
            nodata = self.meta['nodata']
            meta = self.meta
            if callable(self.distance_limit):
                raster[~self.distance_limit(raster)] = np.nan

            raster[raster == nodata] = np.nan
            min_value = np.nanmin(raster)
            max_value = np.nanmax(raster)
            raster = (raster - min_value) / (max_value - min_value)
            if inverse:
                if not buffer:
                    raster[np.isnan(raster)] = 1
                raster = 1 - raster
            else:
                if not buffer:
                    raster[np.isnan(raster)] = 0

            raster[self.data == nodata] = np.nan
            meta.update(nodata=np.nan, dtype='float32')

            normalized = RasterLayer(category=self.category,
                                     name=self.name + ' - normalized')
            normalized.data = raster
            normalized.meta = meta

            if output_path:
                normalized.save(output_path)

            if create_raster:
                self.normalized = normalized
            else:
                return normalized

    def polygonize(self) -> VectorLayer:
        """Polygonizes the raster layer based on the gridded data values.

        It takes the unique values from the ``data`` array as categories and converts the cells into polygons.

        Returns
        ----------
        VectorLayer
            :class:`VectorLayer` with the polygonized data.
        """
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(features.shapes(self.data, transform=self.meta['transform'])))

        geoms = list(results)
        polygon = gpd.GeoDataFrame.from_features(geoms)
        polygon.crs = self.meta['crs']
        return polygon

    def save(self, output_path: str):
        """Saves the raster layer as a `tif` file.

        It uses the ``name`` attribute as the name of the file.

        Parameters
        ----------
        output_path: str
            A folder path where to save the output dataset.
        """
        output_file = os.path.join(output_path,
                                   self.name + '.tif')
        self.path = output_file
        os.makedirs(output_path, exist_ok=True)
        self.meta.update(compress='DEFLATE', driver='GTiff')
        with rasterio.open(output_file, "w", **self.meta) as dest:
            dest.write(self.data, 1)

    def align(self, base_layer: Union['RasterLayer', str],
              rescale: Optional[bool] = None,
              output_path: Optional[str] = None,
              inplace: bool = True) -> 'RasterLayer':
        """Aligns the rasters gridded data with grid of an input raster.

        Parameters
        ----------
        base_layer: RasterLater or str path to file
            Raster layer to use as base for the grid alignment.
        rescale: bool, optional
            Whether to rescale the values proportionally to the the cell size difference between the
            ``base_layer`` and the current raster. If not defined then the ``rescale`` attribute is used.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the aligned raster is not saved
            to disk.
        inplace: bool, default True
            Whether to replace the current raster with the aligned data or return a new :class:`RasterLayer`.

        Returns
        -------
        RasterLayer
            :class:`RasterLayer` with the aligned data.
        """
        if isinstance(base_layer, str):
            with rasterio.open(base_layer) as src:
                base_layer = RasterLayer(path=base_layer)

        if rescale is None:
            rescale = self.rescale

        if rescale:
            crs = base_layer.meta['crs']
            cell_size = base_layer.meta['transform'][0]
            transform = self.calculate_default_transform(crs)[0]

        layer, meta = align_raster(base_layer, self,
                                   method=self.resample)
        data = layer
        meta = meta

        if rescale:
            data[data == meta['nodata']] = np.nan
            meta['nodata'] = np.nan
            factor = (cell_size ** 2) / (transform[0] ** 2)
            data *= factor
        if output_path is not None:
            self.save(output_path)
        if inplace:
            self.data = data
            self.meta = meta
        else:
            raster = RasterLayer()
            raster.data = data
            raster.meta = meta
            return raster

    def cumulative_count(self, min_max: list[float, float] = [0.02, 0.98]) -> np.ndarray:
        """Calculates a new data array flattening the raster's data values that fall in in either of the lower or upper
        specified percentile.

        For example, a if we use a ``min_max`` of ``[0.02, 0.98]``, the array will be first ordered in ascending
        order and then all values that fall inside the lowest 2% will be "flattened" giving them the value of the
        highest number inside that 2%. The same is done for the upper bound, where all data values that fall in the
        highest 2% will be given the value of the lowest number within that 2%.

        Parameters
        ----------
        min_max: list of float
           List of lower and upper limits to consider for the cumulative count.

        Returns
        -------
        np.ndarray
           Raster data array with the flattened values.
        """
        x = self.data.astype('float64').copy()
        x[x == self.meta['nodata']] = np.nan
        x = x.flat
        x = np.sort(x[~np.isnan(x)])
        count = x.shape[0]
        max_val = x[int(count * min_max[1])]
        min_val = x[int(count * min_max[0])]
        layer = self.data.copy()
        layer[layer == self.meta['nodata']] = np.nan
        layer[layer > max_val] = max_val
        layer[layer < min_val] = min_val
        return layer

    def get_quantiles(self, quantiles: tuple[float]) -> np.ndarray:
        """Gets the values of th specified quantiles.

        It uses the :doc:`numpy:reference/generated/numpy.quantile` function to return the quantiles of
        the raster array.

        Parameters
        ----------
        quantiles: array_like of float
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

        Returns
        -------
        np.ndarray
            Quantile values of the raster array.

        Notes
        -----
        Refer to :doc:`numpy:reference/generated/numpy.quantile` for more information.
        """
        x = self.data.astype('float64').copy()
        x[x == self.meta['nodata']] = np.nan
        x = x.flat
        x = x[~np.isnan(x)].copy()
        return np.quantile(x, quantiles)

    def quantiles(self, quantiles: tuple[float]) -> np.ndarray:
        """Computes an array based on the desired quantiles of the raster array.

        It creates an array with the quantile categories of the raster array. Uses the :meth:`get_quantiles` method.

        Parameters
        ----------
        quantiles: array-like of float
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

        Returns
        -------
        np.ndarray
            Categorized array based on the quantile values of the raster array.

        See also
        --------
        get_quantiles
        """
        qs = self.get_quantiles(quantiles)
        layer = self.data.copy()
        layer[layer == self.meta['nodata']] = np.nan
        i = 0
        min_val = np.nanmin(layer)
        layer = layer - min_val
        qs = qs - min_val
        new_layer = layer.copy()
        while i < (len(qs)):
            if i == 0:
                new_layer[(layer >= 0) & (layer < qs[i])] = 0
            else:
                new_layer[(layer >= qs[i - 1]) & (layer < qs[i])] = quantiles[i - 1] * 100
            i += 1
        # if quantiles[i - 1] >= 1:
        #     new_layer[layer >= qs[i - 2]] = 100
        # else:
        new_layer[(layer >= qs[i - 2]) & (layer <= qs[i - 1])] = quantiles[i - 2] * 100
        new_layer[layer > qs[i - 1]] = np.nan
        return new_layer

    @staticmethod
    def category_legend(im, ax, categories, current_handles_labels=None,
                        legend_position=(1.05, 1), title='', legend_cols=1,
                        legend_prop={'title': {'size': 12, 'weight': 'bold'}, 'size': 12}):
        """Creates a category legend for the current plot.

        Parameters
        ----------
        im
        categories
        legend_position
        title
        legend_cols
        legend_prop
        """
        values = list(categories.values())
        titles = list(categories.keys())

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label="{}".format(titles[i])) for i in range(len(values))]
        # put those patched as legend-handles into the legend
        prop = legend_prop.copy()
        prop.pop('title')
        # ax.legend(handles=patches)
        # if current_handles_labels is not None:
        #     new_handles = ax.get_legend().legendHandles
        #     new_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        #     handles = current_handles_labels[0] + new_handles
        #     labels = current_handles_labels[1] + new_labels
        legend = ax.legend(handles=patches, bbox_to_anchor=legend_position, loc='upper left',
                           borderaxespad=0., ncol=legend_cols, prop=prop)
        legend.set_title(title, prop=legend_prop['title'])
        legend._legend_box.align = "left"
        ax.add_artist(legend)


    def plot(self, cmap='viridis', ticks=None, tick_labels=None,
             cumulative_count=None, quantiles=None, categories=None, legend_position=(1.05, 1),
             admin_layer: Union[gpd.GeoDataFrame, VectorLayer] = None, title=None, ax=None, dpi=150,
             legend=True, legend_title='', legend_cols=1,
             legend_prop={'title': {'size': 12, 'weight': 'bold'}, 'size': 12},
             kwargs={},
             rasterized=True, colorbar=True, return_image=False, figsize=(6.4, 4.8),
             scale_bar=None, north_arrow=None):
        """Plots a map of the current raster layer.

        Parameters
        ----------
        cmap
        ticks
        tick_labels
        cumulative_count
        quantiles
        categories
        legend_position
        admin_layer
        title
        ax
        dpi
        legend
        legend_title
        legend_cols
        legend_prop
        rasterized
        colorbar
        return_image
        fig_size
        scale_bar
        north_arrow

        Returns
        -------
        matplotlib image
        """

        extent = [self.bounds[0], self.bounds[2],
                  self.bounds[1], self.bounds[3]]  # [left, right, bottom, top]

        if cumulative_count:
            layer = self.cumulative_count(cumulative_count)
        elif quantiles is not None:
            layer = self.quantiles(quantiles)
            qs = self.get_quantiles(quantiles)
            categories = {}
            for i, j in enumerate(quantiles):
                if i == 0:
                    categories[f'$<{int(qs[i])}$'] = j * 100
                elif j >= 1:
                    categories[f'$\geq{int(qs[i - 1])}$'] = j * 100
                elif i < len(qs):
                    categories[f'${int(qs[i - 1])}$' + ' to ' + f'${int(qs[i])}$'] = j * 100
            legend = True
            if legend_title is None:
                legend_title = 'Quantiles'
        else:
            layer = self.data
            layer = layer.astype('float64')
            layer[layer == self.meta['nodata']] = np.nan


        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if isinstance(cmap, dict):
            values = np.sort(np.unique(layer[~np.isnan(layer)]))
            key_list_cat = list(categories.keys())
            val_list_cat = list(categories.values())
            for i, val in enumerate(values):
                layer[layer==val] = i
                position = val_list_cat.index(val)
                categories[key_list_cat[position]] = i
            cmap = ListedColormap([to_rgb(cmap[i]) for i in values])

        if ax.get_legend() is not None:
            if ax.get_legend().legendHandles is not None:
                handles = ax.get_legend().legendHandles
                labels = [t.get_text() for t in ax.get_legend().get_texts()]
                ax.legend(handles=handles, labels=labels)
        else:
            handles = []
            labels = []

        cax = ax.imshow(layer, cmap=cmap, extent=extent, interpolation='none', zorder=1, rasterized=rasterized,
                        **kwargs)

        if legend:
            if categories:
                self.category_legend(cax, ax, categories, current_handles_labels=(handles, labels),
                                     legend_position=legend_position,
                                     title=legend_title, legend_cols=legend_cols, legend_prop=legend_prop)
            elif colorbar:
                colorbar = dict(shrink=0.8)
                if ticks:
                    colorbar['ticks'] = ticks
                cbar = plt.colorbar(cax, **colorbar)

                if tick_labels:
                    cbar.ax.set_yticklabels(tick_labels)
                cbar.ax.set_ylabel(self.name.replace('_', ' '))
        if isinstance(admin_layer, VectorLayer):
            admin_layer = admin_layer.data
        if isinstance(admin_layer, gpd.GeoDataFrame):
            if admin_layer.crs != self.meta['crs']:
                admin_layer.to_crs(self.meta['crs'], inplace=True)
            admin_layer.plot(color=to_rgb('#f1f1f1ff'), linewidth=1, ax=ax, zorder=0, rasterized=rasterized)
        if title:
            plt.title(title, loc='left')

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if scale_bar is not None:
            if scale_bar == 'default':
                scale_bar_func()
            elif isinstance(scale_bar, dict):
                scale_bar_func(**scale_bar)
            else:
                raise ValueError('Parameter `scale_bar` need to be a dictionary with parameter/value pairs, '
                                 'accepted by the `onstove.scale_bar` function.')

        if north_arrow is not None:
            if north_arrow == 'default':
                north_arrow_func()
            elif isinstance(north_arrow, dict):
                north_arrow_func(**north_arrow)
            else:
                raise ValueError('Parameter `north_arrow` need to be a dictionary with parameter/value pairs, '
                                 'accepted by the `onstove.north_arrow` function.')

        return ax

    def save_image(self, output_path, type='png', cmap='viridis', ticks=None, tick_labels=None,
                   cumulative_count=None, categories=None, legend_position=(1.05, 1), figsize=(6.4, 4.8),
                   admin_layer=None, title=None, ax=None, dpi=300, quantiles=None,
                   legend_prop={'title': {'size': 12, 'weight': 'bold'}, 'size': 12},
                   legend=True, legend_title='', legend_cols=1, rasterized=True, scale_bar=None, north_arrow=None):
        """Saves the raster as an image map in the specified format.
        """
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path,
                                   self.name + f'.{type}')
        self.plot(cmap=cmap, ticks=ticks, tick_labels=tick_labels, cumulative_count=cumulative_count,
                  categories=categories, legend_position=legend_position, rasterized=rasterized,
                  admin_layer=admin_layer, title=title, ax=ax, dpi=dpi, quantiles=quantiles,
                  legend=legend, legend_title=legend_title, legend_cols=legend_cols,
                  scale_bar=scale_bar, north_arrow=north_arrow, figsize=figsize, legend_prop=legend_prop)

        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()

    def save_style(self, output_path, cmap='magma', quantiles=None,
                   categories=None, classes=5):
        """Saves the colormap used for the raster as a sld style."""
        if categories is not None:
            colors = cmap
        else:
            colors = plt.get_cmap(cmap, classes).colors

        if quantiles:
            qs = self.get_quantiles(quantiles)
            colors = plt.get_cmap(cmap, len(qs)).colors
        else:
            qs = np.linspace(np.nanmin(self.data), np.nanmax(self.data), num=len(colors))

        if categories is None:
            categories = {i: round(i, 2) for i in qs}

        values = """"""
        for i, value in enumerate(qs):
            values += f"""\n{" " * 14}<sld:ColorMapEntry label="{categories[value]}" color="{to_hex(colors[i])}" quantity="{value}"/>"""

        string = f"""<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld" xmlns:gml="http://www.opengis.net/gml">
  <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>{self.name}</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ChannelSelection>
              <sld:GrayChannel>
                <sld:SourceChannelName>1</sld:SourceChannelName>
              </sld:GrayChannel>
            </sld:ChannelSelection>
            <sld:ColorMap type="ramp">{values}
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
"""
        with open(os.path.join(output_path, f'{self.name}.sld'), 'w') as f:
            f.write(string)
