"""This module contains the GIS layer classes used in OnStove."""
import math
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
from rasterio import windows
from rasterio.transform import array_bounds
from rasterio import warp, features
from matplotlib.colors import ListedColormap, to_rgb, to_hex
from scipy import ndimage
from typing import Optional, Callable, Union

from onstove.raster import normalize, reproject_raster, align_raster


def try_import():
    try:
        from skimage.graph.mcp import MCP_Geometric
        return MCP_Geometric
    except Exception as e:
        print('Trying import again...')
        time.sleep(1)
        return try_import()


MCP_Geometric = try_import()


class Layer:
    """
    Template Layer initializing all common needed attributes.
    """

    def __init__(self, category: Optional[str] = None, name: Optional[str] = None,
                 path: Optional[str] = None, conn: Optional[str] = None,
                 normalization: Optional[str] = 'MinMax', inverse: bool = False,
                 distance:  Optional[str] = 'proximity', distance_limit: Optional[float] = None):
        self.category = category
        self.name = name
        self.normalization = normalization
        self.distance = distance
        self.distance_limit = distance_limit
        self.inverse = inverse
        self.friction = None
        self.distance_raster = None
        self.restrictions = []
        self.weight = 1
        self.path = path

    def __repr__(self):
        return 'Layer(name=%r)' % self.name

    def __str__(self):
        s = f'Layer\n    - Name: {self.name}\n'
        for attr, value in self.__dict__.items():
            s += f'    - {attr}: {value}\n'
        return s

    def read_layer(self, layer_path, conn=None):
        pass


class VectorLayer(Layer):
    """
    A ``VectorLayer`` is an object used to read, manipulate and visualize GIS vector data.

    It uses a :doc:`GeoDataFrame<geopandas:docs/reference/geodataframe>` object  to store the georeferenced data in a
    tabular format. It also stores metadata as `category` and `name` of the layer, `normalization` and `distance`
    algorithms to use among others. This data structure is used in both the ``MCA`` and the
    :class:`OnStove<onstove.onstove.OnStove>` models and the :class:`OnStove<onstove.onstove.DataProcessor>` object.

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
            :meth:`read_layer` and :meth:`onstove.onstove.DataProcessor.set_postgres`

    query: str, optional
        A query string to filter the data. For more information refer to
        :doc:`pandas:reference/api/pandas.DataFrame.query`.

        .. seealso::
            :meth:`read_layer`

    normalization: str, default 'MinMax'
        Sets the default normalization method to use when calling the :meth:`RasterLayer.normalize` for any
        associated :class:`RasterLayer` (for example a distance raster). This is relevant to calculate the
        :attr:`demand_index<onstove.onstove.DataProcessor.demand_index>`,
        :attr:`supply_index<onstove.onstove.DataProcessor.supply_index>`,
        :attr:`clean_cooking_index<onstove.onstove.DataProcessor.clean_cooking_index>` and
        :attr:`assistance_need_index<onstove.onstove.DataProcessor.assistance_need_index>` of the ``MCA`` model.
    inverse: str, optional
        Sets the default mode for the normalization algorithm (see :meth:`RasterLayer.normalize`).
    distance: str, default 'proximity'
        Sets the default distance algorithm to use when calling the :meth:`get_distance_raster` method.
    distance_limit: Callable object (function or lambda function) with a numpy array as input, optional
        Defines a distance limit or range to consider when calculating the distance raster.

        .. code-block:: python
           :caption: Example: lambda function for distance range between 1,000 and 10,000 meters

           distance_limit = lambda  x: (x >= 1000) & (x <= 10000)

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
        list of :class:`RasterLayer` or :class:`VectorLayer` used to restrict areas from the distance calculations.
    weight
        Value to weigh the layer's "importance" on the ``MCA`` model. It is initialized with a default value of 1.
    bounds
    style: dict
        Contains the default style parameters for visualizing the layer.

        .. seealso:
            :meth:`plot`
    """

    def __init__(self, category: Optional[str] = None, name: Optional[str] = None,
                 path: Optional[str] = None, conn: Optional[str] = None, query: Optional[str] = None,
                 normalization: Optional[str] = 'MinMax', inverse: bool = False,
                 distance:  Optional[str] = 'proximity',
                 distance_limit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 bbox: Optional[gpd.GeoDataFrame] = None):
        """
        Initializes the class with user defined or default parameters.
        """
        self.style = {}
        super().__init__(category=category, name=name,
                         path=path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance=distance,
                         distance_limit=distance_limit)
        self.read_layer(path=path, conn=conn, bbox=bbox)
        if query:
            self.layer = self.layer.query(query)

    def __repr__(self):
        return 'Vector' + super().__repr__()

    def __str__(self):
        return 'Vector' + super().__str__()

    @property
    def bounds(self) -> list[float]:
        """Wrapper property to get the bounds of the dataset using the total_bounds property of Pandas.
        """
        return self.layer['geometry'].total_bounds

    @property
    def friction(self):
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
        else:
            raise ValueError('Raster file type or object not recognized.')

    def read_layer(self, path, conn=None, bbox=None):
        """Reads a dataset from GIS vector data file.

        It works as a wrapper method that will use either the :doc:`geopandas:docs/reference/api/geopandas.read_file`
        function or the :doc:`geopandas:docs/reference/api/geopandas.read_postgis` function to read vector data and
        store the output in the :attr:`layer` attribute and the layer path in the ``path`` attribute.

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
        """
        if path:
            if conn:
                sql = f'SELECT * FROM {path}'
                self.layer = gpd.read_postgis(sql, conn)
            else:
                if isinstance(bbox, gpd.GeoDataFrame):
                    bbox = bbox.dissolve()
                elif bbox is not None:
                    raise ValueError('The `bbox` parameter should be of type GeoDataFrame or None, '
                                     f'type {type(bbox)} was given')
                self.layer = gpd.read_file(path, bbox=bbox)
        self.path = path

    def mask(self, mask_layer: gpd.GeoDataFrame, output_path: str = None):
        """Wrapper for the :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame.clip` method.

        Parameters
        ----------
        mask_layer: GeoDataFrame
            A :doc:`geopandas:docs/reference/api/geopandas.GeoDataFrame` object.
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the clipped dataset is not saved.
        """
        self.layer = gpd.clip(self.layer, mask_layer.to_crs(self.layer.crs))
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
        if self.layer.crs != crs:
            self.layer.to_crs(crs, inplace=True)
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
                bounds = src.bounds
                width = src.width
                height = src.height
                transform = src.transform
        elif isinstance(base_layer, RasterLayer):
            bounds = base_layer.bounds
            width = base_layer.meta['width']
            height = base_layer.meta['height']
            transform = base_layer.meta['transform']
        else:
            raise ValueError('The `base_layer` (or `raster` if you are using the `get_distance_raster` method) '
                             'must be either a raster file path or a '
                             f'RasterLayer object. {type(base_layer)} was given instead.')

        data, meta = self.rasterize(value=1, width=width, height=height,
                                    transform=transform)

        data = ndimage.distance_transform_edt(1 - data.astype(int),
                                              sampling=[meta['transform'][0], -meta['transform'][4]])

        meta.update(nodata=np.nan, dtype='float32')

        distance_raster = RasterLayer(self.category,
                                      self.name + '_dist',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)
        distance_raster.layer = data
        distance_raster.meta = meta
        distance_raster.bounds = bounds

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
            attribute. If None is used, then the predefined ``distance`` attribute of the :class:`VectorLayer` class
            is used instead.
        raster: RasterLayer
            Raster layer used as base or friction layer depending on the ``method`` used (see above).
        output_path: str, optional
            A folder path where to save the output dataset. If not defined then the distance raster dataset is not
            saved.
        """
        if method is None:
            if self.distance is None:
                raise ValueError('Please pass a distance `method` ("proximity" or "travel time") or define the default '
                                 'method in the `distance` attribute of the class.')
            method = self.distance

        if method == 'proximity':
            self.proximity(base_layer=raster, output_path=output_path, create_raster=True)

        elif method == 'travel_time':
            self.travel_time(friction=raster, output_path=output_path, create_raster=True)

    def rasterize(self, attribute: str = None,
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
        bounds = self.layer['geometry'].total_bounds
        if transform is None:
            if (width is None) or (height is None):
                width, height = RasterLayer.shape_from_cell(bounds, cell_height, cell_width)
            transform = rasterio.transform.from_bounds(*bounds, width, height)
        else:
            width, height = RasterLayer.shape_from_cell(bounds, transform[0], -transform[4])

        if attribute:
            shapes = ((g, v) for v, g in zip(self.layer[attribute].values, self.layer['geometry'].values))
            dtype = type(self.layer[attribute].values[0])
        else:
            shapes = ((g, value) for g in self.layer['geometry'].values)
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
                    crs=self.layer.crs,
                    width=width,
                    height=height,
                    transform=transform,
                    nodata=nodata)
        raster = RasterLayer()
        raster.layer = rasterized
        raster.meta = meta
        raster.bounds = bounds

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
        for index, row in self.layer.iterrows():
            rows, cols = rasterio.transform.rowcol(raster.meta['transform'], row["geometry"].x, row["geometry"].y)
            row_list.append(rows)
            col_list.append(cols)

        return row_list, col_list

    def save(self, output_path: str):
        """Saves the current :class:`VectorLayer` into disk.

        It saves the layer in the ``output_path`` defined using the ``name`` attribute as filename and `geojson` as
        extension.

        Parameters
        ----------
        output_path: str
            Output folder where to save the layer.
        """
        for column in self.layer.columns:
            if isinstance(self.layer[column].iloc[0], datetime.date):
                self.layer[column] = self.layer[column].astype('datetime64')
        output_file = os.path.join(output_path,
                                   self.name + '.geojson')
        os.makedirs(output_path, exist_ok=True)
        self.layer.to_file(output_file, driver='GeoJSON')
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

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None, style: dict = None):
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
        if style is None:
            style = self.style

        if ax is None:
            ax = self.layer.plot(**style,
                                 label=self.name)
            # lgnd = ax.legend(loc="upper right", prop={'size': 12})
            # lgnd.legendHandles[0]._sizes = [60]
        else:
            self.layer.plot(ax=ax,
                            **style,
                            label=self.name)


class RasterLayer(Layer):
    """
    Layer class for GIS Raste data. It stores a Numpy array and the metadata 
    of a rasterio object. Also some extra metadata is stored as layer name and 
    normalization algorithm.
    """

    def __init__(self, category='', name='', path=None, conn=None,
                 normalization='MinMax', inverse=False, distance='proximity',
                 distance_limit=None, resample='nearest', window=None,
                 rescale=False):
        """
        Initializes the class. It receives the name of the layer,
        the path of the layer, a normalization algorithm, a distance algorithm
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.resample = resample
        self.rescale = rescale
        super().__init__(category=category, name=name,
                         path=path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance=distance,
                         distance_limit=distance_limit)
        self.read_layer(path, conn, window=window)

    def __repr__(self):
        return 'Raster' + super().__repr__()

    def __str__(self):
        return 'Raster' + super().__str__()

    def read_layer(self, layer_path, conn=None, window=None):
        if layer_path:
            with rasterio.open(layer_path) as src:
                if window:
                    transform = src.transform
                    self.meta = src.meta.copy()
                    self.bounds = window
                    window = windows.from_bounds(*window, transform=transform)
                    window_transform = src.window_transform(window)
                    self.layer = src.read(1, window=window)
                    self.meta.update(transform=window_transform,
                                     width=self.layer.shape[1], height=self.layer.shape[0],
                                     compress='DEFLATE')
                else:
                    self.layer = src.read(1)
                    self.meta = src.meta
                    self.bounds = src.bounds
                    # self.meta['dtype'] = 'float32'
            if self.meta['nodata'] is None:
                self.meta['nodata'] = np.nan
        self.path = layer_path

    @staticmethod
    def shape_from_cell(bounds, cell_height, cell_width):
        height = round((bounds[3] - bounds[1]) / cell_height)
        width = round((bounds[2] - bounds[0]) / cell_width)
        return width, height

    def mask(self, mask_layer, output_path=None, crop=True, all_touched=True):
        shape_mask, meta = mask_layer.rasterize(value=1, transform=self.meta['transform'],
                                                width=self.meta['width'], height=self.meta['height'],
                                                nodata=0, all_touched=all_touched)

        self.layer[shape_mask == 0] = self.meta['nodata']

        if crop:
            total_bounds = mask_layer.layer['geometry'].total_bounds
            window = windows.from_bounds(*total_bounds, transform=self.meta['transform'])
            width, height = self.shape_from_cell(total_bounds, self.meta['transform'][0], -self.meta['transform'][4])
            row_off = max(round(window.row_off), 0)
            col_off = max(round(window.col_off), 0)
            window = windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=width,
                height=height,
            )
            bounds = rasterio.windows.bounds(window, self.meta['transform'])
            transform = rasterio.transform.from_bounds(*bounds, width, height)
            self.layer = self.layer[row_off:(row_off + height),
                                    col_off:(col_off + width)]
            self.meta.update(transform=transform, height=height, width=width)

        if output_path:
            self.save(output_path)

        # output_file = os.path.join(output_path,
        #                            self.name + '.tif')
        # mask_raster(self.path, mask_layer.to_crs(self.meta['crs']),
        #             output_file, self.meta['nodata'], 'DEFLATE', all_touched=all_touched)
        # self.read_layer(output_file)

    def reproject(self, crs, output_path=None,
                  cell_width=None, cell_height=None):
        if (self.meta['crs'] != crs) or cell_width:
            data, meta = reproject_raster(self.path, crs,
                                          cell_width=cell_width, cell_height=cell_height,
                                          method=self.resample, compression='DEFLATE')
            self.layer = data
            self.bounds = warp.transform_bounds(self.meta['crs'], crs, *self.bounds)
            self.meta = meta
            if output_path:
                self.save(output_path)

    def calculate_default_transform(self, dst_crs):
        t, w, h = warp.calculate_default_transform(self.meta['crs'],
                                                   dst_crs,
                                                   self.meta['width'],
                                                   self.meta['height'],
                                                   *self.bounds,
                                                   dst_width=self.meta['width'],
                                                   dst_height=self.meta['height'])
        return t, w, h

    def travel_time(self, rows, cols,
                    output_path: Optional[str] = None,
                    create_raster: Optional[bool] = True) -> 'RasterLayer':
        layer = self.layer.copy()
        layer *= (1000 / 60)  # to convert to hours per kilometer
        layer[np.isnan(layer)] = float('inf')
        layer[layer == self.meta['nodata']] = float('inf')
        layer[layer < 0] = float('inf')
        mcp = MCP_Geometric(layer, fully_connected=True)
        pointlist = np.column_stack((rows, cols))
        # TODO: create method for restricted areas
        if len(pointlist) > 0:
            cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
            cumulative_costs[np.where(cumulative_costs == float('inf'))] = np.nan
        else:
            cumulative_costs = np.full(self.layer.shape, 2.0)

        distance_raster = RasterLayer(self.category,
                                      'traveltime',
                                      distance_limit=self.distance_limit,
                                      inverse=self.inverse,
                                      normalization=self.normalization)

        meta = self.meta.copy()
        meta.update(nodata=np.nan)
        distance_raster.layer = cumulative_costs  # + (self.friction.layer * 1000 / 60)
        distance_raster.meta = meta
        distance_raster.bounds = self.friction.bounds

        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster

    def log(self, mask_layer,
            output_path: Optional[str] = None,
            create_raster: Optional[bool] = True) -> 'RasterLayer':
        layer = self.layer.copy()
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
        distance_raster.layer = layer
        distance_raster.meta = meta
        # distance_raster.save(output_path)
        distance_raster.mask(mask_layer)

        if output_path:
            distance_raster.save(output_path)

        if create_raster:
            self.distance_raster = distance_raster
        else:
            return distance_raster

    def get_distance_raster(self, output_path, mask_layer, starting_points=None):
        if self.distance == 'log':
            self.distance_raster = self.log(mask_layer=mask_layer)
        elif self.distance == 'travel_time':
            rows, cols = self.start_points(condition=starting_points)
            self.distance_raster = self.travel_time(rows, cols)
        else:
            self.distance_raster = self

    def start_points(self, condition=None):
        if callable(condition):
            return np.where(condition(self.layer))
        else:
            return np.where(np.isin(self.layer, self.starting_cells))

    def normalize(self, output_path, mask_layer=None, buffer=False):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path,
                                       self.name + ' - normalized.tif')
            normalize(raster=self.layer, limit=self.distance_limit,
                      inverse=self.inverse, output_file=output_file,
                      meta=self.meta, buffer=buffer)
            self.normalized = RasterLayer(self.category, self.name + ' - normalized',
                                          path=output_file)
            self.mask(mask_layer=mask_layer, output_path=output_file, all_touched=True)

    def polygonize(self):
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v)
            in enumerate(features.shapes(self.layer, transform=self.meta['transform'])))

        geoms = list(results)
        polygon = gpd.GeoDataFrame.from_features(geoms)
        polygon.crs = self.meta['crs']
        return polygon

    def save(self, output_path, sufix=''):
        output_file = os.path.join(output_path,
                                   self.name + f'{sufix}.tif')
        self.path = output_file
        os.makedirs(output_path, exist_ok=True)
        self.meta.update(compress='DEFLATE')
        with rasterio.open(output_file, "w", **self.meta) as dest:
            dest.write(self.layer, indexes=1)

    def add_friction_raster(self, raster, starting_cells=[1],
                            resample='nearest'):
        self.starting_cells = starting_cells
        if isinstance(raster, str):
            self.friction = RasterLayer(self.category,
                                        self.name + ' - friction',
                                        raster, resample=resample)
        elif isinstance(raster, RasterLayer):
            self.friction = raster
        else:
            raise ValueError('Raster file type or object not recognized.')

    def align(self, base_layer, output_path=None):
        if self.rescale:
            with rasterio.open(base_layer) as src:
                crs = src.meta['crs']
                cell_size = src.meta['transform'][0]
            transform = self.calculate_default_transform(crs)[0]

        layer, meta = align_raster(base_layer, self.path,
                                   method=self.resample)
        self.layer = layer
        self.meta = meta
        self.bounds = array_bounds(meta['height'], meta['width'], meta['transform'])

        if self.rescale:
            self.layer[self.layer == self.meta['nodata']] = np.nan
            self.meta['nodata'] = np.nan
            factor = (cell_size ** 2) / (transform[0] ** 2)
            self.layer *= factor

        if output_path:
            self.save(output_path)

    def cumulative_count(self, min_max=[0.02, 0.98]):
        x = self.layer.flat
        x = np.sort(x[~np.isnan(x)])
        count = x.shape[0]
        max_val = x[int(count * min_max[1])]
        min_val = x[int(count * min_max[0])]
        layer = self.layer.copy()
        layer[layer > max_val] = max_val
        layer[layer < min_val] = min_val
        return layer

    def get_quantiles(self, quantiles):
        x = self.layer.flat
        x = x[~np.isnan(x)].copy()
        return np.quantile(x, quantiles)

    def quantiles(self, quantiles):
        qs = self.get_quantiles(quantiles)
        layer = self.layer.copy()
        i = 0
        min_val = np.nanmin(layer)
        layer = layer - min_val
        qs = qs - min_val
        while i < (len(qs)):
            if i == 0:
                layer[(layer >= 0) & (layer <= qs[i])] = quantiles[i] * 100
            else:
                layer[(layer > qs[i - 1]) & (layer <= qs[i])] = quantiles[i] * 100
            i += 1
        return layer

    @staticmethod
    def category_legend(im, categories, legend_position=(1.05, 1), title='', legend_cols=1):
        values = list(categories.values())
        titles = list(categories.keys())

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label="{}".format(titles[i])) for i in range(len(values))]
        # put those patched as legend-handles into the legend
        legend = plt.legend(handles=patches, bbox_to_anchor=legend_position, loc=2, borderaxespad=0., ncol=legend_cols)
        legend.set_title(title, prop={'size': 12, 'weight': 'bold'})
        legend._legend_box.align = "left"

    def plot(self, cmap='viridis', ticks=None, tick_labels=None,
             cumulative_count=None, quantiles=None, categories=None, legend_position=(1.05, 1),
             admin_layer=None, title=None, ax=None, dpi=150, figsize=(16, 9), legend=True, legend_title='',
             legend_cols=1, rasterized=True, colorbar=True, return_image=False):
        extent = [self.bounds[0], self.bounds[2],
                  self.bounds[1], self.bounds[3]]  # [left, right, bottom, top]

        if cumulative_count:
            layer = self.cumulative_count(cumulative_count)
        elif quantiles:
            layer = self.quantiles(quantiles)
            qs = self.get_quantiles(quantiles)
            categories = {i: j * 100 for i, j in zip(qs, quantiles)}
            legend = True
            legend_title = 'Quantiles'
        else:
            layer = self.layer

        layer = layer.astype('float64')

        layer[layer == self.meta['nodata']] = np.nan

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if isinstance(cmap, dict):
            values = np.sort(np.unique(layer[~np.isnan(layer)]))
            cmap = ListedColormap([to_rgb(cmap[i]) for i in values])
        cax = ax.imshow(layer, cmap=cmap, extent=extent, interpolation='none', zorder=1, rasterized=rasterized)

        if legend:
            if categories:
                self.category_legend(cax, categories, legend_position=legend_position,
                                     title=legend_title, legend_cols=legend_cols)
            elif colorbar:
                colorbar = dict(shrink=0.8)
                if ticks:
                    colorbar['ticks'] = ticks
                cbar = plt.colorbar(cax, **colorbar)

                if tick_labels:
                    cbar.ax.set_yticklabels(tick_labels)
                cbar.ax.set_ylabel(self.name.replace('_', ' '))
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

        if return_image:
            return cax

    def save_image(self, output_path, type='png', cmap='viridis', ticks=None, tick_labels=None,
                   cumulative_count=None, categories=None, legend_position=(1.05, 1),
                   admin_layer=None, title=None, ax=None, dpi=300, quantiles=None,
                   legend=True, legend_title='', legend_cols=1, rasterized=True):
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path,
                                   self.name + f'.{type}')
        self.plot(cmap=cmap, ticks=ticks, tick_labels=tick_labels, cumulative_count=cumulative_count,
                  categories=categories, legend_position=legend_position, rasterized=rasterized,
                  admin_layer=admin_layer, title=title, ax=ax, dpi=dpi, quantiles=quantiles,
                  legend=legend, legend_title=legend_title, legend_cols=legend_cols)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()

    def save_style(self, output_path, cmap='magma', quantiles=None,
                   categories=None, classes=5):
        if categories is not None:
            colors = cmap
        else:
            colors = plt.get_cmap(cmap, classes).colors

        if quantiles:
            qs = self.get_quantiles(quantiles)
            colors = plt.get_cmap(cmap, len(qs)).colors
        else:
            qs = np.linspace(np.nanmin(self.layer), np.nanmax(self.layer), num=len(colors))

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
