import os
import geopandas as gpd
import re
import pandas as pd
import datetime

from rasterio import windows
from rasterio.warp import calculate_default_transform
from skimage.graph.mcp import MCP_Geometric
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .raster import *


class Layer:
    """
    Template Layer initializing all needed variables.
    """

    def __init__(self, category='', name='', layer_path=None,
                 conn=None, normalization=None,
                 inverse=False, distance=None,
                 distance_limit=float('inf')):
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

    def __repr__(self):
        return 'Layer(name=%r)' % self.name

    def __str__(self):
        return f'Layer\n    - Name: {self.name}\n' + \
               f'    - Category: {self.category}\n' + \
               f'    - Normalization: {self.normalization}\n' + \
               f'    - Distance method: {self.distance}\n' + \
               f'    - Distance limit: {self.distance_limit}\n' + \
               f'    - Inverse: {self.inverse}\n' + \
               f'    - Resample: {self.resample}\n' + \
               f'    - Path: {self.path}'

    def read_layer(self, layer_path, conn=None):
        pass

    def travel_time(self, output_path, condition=None):
        layer = self.friction.layer.copy()
        layer *= 1000 / 60  # to convert to hours per kilometer
        layer[np.isnan(layer)] = float('inf')
        mcp = MCP_Geometric(layer, fully_connected=True)
        row, col = self.start_points(condition=condition)
        pointlist = np.column_stack((row, col))
        # TODO: create method for restricted areas
        cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
        cumulative_costs[np.where(cumulative_costs == float('inf'))] = np.nan

        self.distance_raster = RasterLayer(self.category,
                                           self.name + ' - traveltime',
                                           distance_limit=self.distance_limit,
                                           inverse=self.inverse,
                                           normalization=self.normalization)

        self.distance_raster.layer = cumulative_costs
        self.distance_raster.meta = self.friction.meta.copy()
        self.distance_raster.bounds = self.friction.bounds
        self.distance_raster.save(output_path)


class VectorLayer(Layer):
    """
    Layer class for GIS Vector data. It stores a GeoPandas dataframe and some
    required metadata as layer name, normalization algorithm and 
    distance algorithm.
    """

    def __init__(self, category, name, layer_path=None, conn=None, query=None,
                 normalization='MinMax', inverse=False, distance='proximity',
                 distance_limit=float('inf')):
        """
        Initializes the class. It recibes the name of the layer, 
        the path of the layer, a normalization algorithm, a distance algorithm 
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        super().__init__(category=category, name=name,
                         layer_path=layer_path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance=distance,
                         distance_limit=distance_limit)
        self.read_layer(layer_path, conn)
        if query:
            self.layer = self.layer.query(query)

    def __repr__(self):
        return 'Vector' + super().__repr__()

    def __str__(self):
        return 'Vector' + super().__str__()

    def read_layer(self, layer_path, conn=None):
        if layer_path:
            if conn:
                sql = f'SELECT * FROM {layer_path}'
                self.layer = gpd.read_postgis(sql, conn)
            else:
                self.layer = gpd.read_file(layer_path)
        self.path = layer_path

    def mask(self, mask_layer, output_path, all_touched=True):
        self.layer = gpd.clip(self.layer, mask_layer.to_crs(self.layer.crs))
        self.save(output_path)

    def reproject(self, crs, output_path):
        if self.layer.crs != crs:
            self.layer.to_crs(crs, inplace=True)
            self.save(output_path)

    def get_distance_raster(self, base_layer, output_path,
                            mask_layer):
        if self.distance == 'proximity':
            output_rasterized = os.path.join(output_path,
                                             self.name + ' - rasterized.tif')
            rasterize(self.layer, base_layer,
                      output_rasterized, compression='DEFLATE',
                      nodata=0, save=True)
            output_proximity_temp = os.path.join(output_path,
                                                 self.name + ' - proximity_temp.tif')
            proximity_raster(output_rasterized,
                             output_proximity_temp,
                             [1], 'DEFLATE')
            output_proximity = os.path.join(output_path,
                                            self.name + '_dist.tif')
            mask_raster(output_proximity_temp, mask_layer,
                        output_proximity, np.nan, 'DEFLATE')
            os.remove(output_rasterized)
            os.remove(output_proximity_temp)
            self.distance_raster = RasterLayer(self.category,
                                               self.name + '_dist',
                                               output_proximity,
                                               distance_limit=self.distance_limit,
                                               inverse=self.inverse,
                                               normalization=self.normalization)

        elif self.distance == 'travel_time':
            self.travel_time(output_path)


    def start_points(self):
        return friction_start_points(self.friction.path,
                                     self.layer)

    def save(self, output_path):
        for column in self.layer.columns:
            if isinstance(self.layer[column].iloc[0], datetime.date):
                self.layer[column] = self.layer[column].astype('datetime64')
        output_file = os.path.join(output_path,
                                   self.name + '.geojson')
        os.makedirs(output_path, exist_ok=True)
        self.layer.to_file(output_file, driver='GeoJSON')
        self.path = output_file

    def add_friction_raster(self, raster, resample='nearest'):
        if isinstance(raster, str):
            self.friction = RasterLayer(self.category, self.name + ' - friction',
                                        raster, resample=resample)
        elif isinstance(raster, RasterLayer):
            self.friction = raster
        else:
            raise ValueError('Raster file type or object not recognized.')

    def add_restricted_areas(self, layer_path, layer_type, **kwargs):
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


class RasterLayer(Layer):
    """
    Layer class for GIS Raste data. It stores a Numpy array and the metadata 
    of a rasterio object. Also some extra metadata is stored as layer name and 
    normalization algorithm.
    """
    def __init__(self, category, name, layer_path=None, conn=None,
                 normalization='MinMax', inverse=False, distance='proximity',
                 distance_limit=float('inf'), resample='nearest', window=None):
        """
        Initializes the class. It recibes the name of the layer,
        the path of the layer, a normalization algorithm, a distance algorithm
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.resample = resample
        super().__init__(category=category, name=name,
                         layer_path=layer_path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance=distance,
                         distance_limit=distance_limit)
        self.read_layer(layer_path, conn, window=window)

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
                    self.layer = src.read(1, window=window).astype('float32')
                    self.meta.update(transform=window_transform,
                                     width=self.layer.shape[1], height=self.layer.shape[0],
                                     compress='DEFLATE', dtype='float32')
                else:
                    self.layer = src.read(1).astype('float32')
                    self.meta = src.meta
                    self.bounds = src.bounds
                    self.meta['dtype'] = 'float32'
        self.path = layer_path

    def mask(self, mask_layer, output_path, all_touched=False):
        output_file = os.path.join(output_path,
                                   self.name + '.tif')
        mask_raster(self.path, mask_layer.to_crs(self.meta['crs']),
                    output_file, np.nan, 'DEFLATE', all_touched=all_touched)
        self.read_layer(output_file)

    def reproject(self, crs, output_path,
                  cell_width=None, cell_height=None):
        if (self.meta['crs'] != crs) or cell_width:
            data, meta = reproject_raster(self.path, crs,
                                          cell_width=cell_width, cell_height=cell_height,
                                          method=self.resample, compression='DEFLATE')
            self.layer = data
            self.meta = meta
            self.save(output_path)

    def calculate_default_transform(self, dst_crs):
        t, w, h = calculate_default_transform(self.meta['crs'],
                                              dst_crs,
                                              self.meta['width'],
                                              self.meta['height'],
                                              *self.bounds)
        return t, w, h

    def get_distance_raster(self, base_layer, output_path,
                            mask_layer):
        if self.distance == 'log':
            layer = self.layer.copy()
            layer[layer == 0] = np.nan
            layer[layer > 0] = np.log(layer[layer > 0])
            layer = np.nan_to_num(layer, nan=0)
            # layer[layer < 0] = np.nan

            meta = self.meta.copy()
            meta.update(nodata=np.nan, dtype='float64')

            self.distance_raster = RasterLayer(self.category,
                                               self.name + ' - log',
                                               distance_limit=self.distance_limit,
                                               inverse=self.inverse,
                                               normalization=self.normalization)

            self.distance_raster.layer = layer
            self.distance_raster.meta = meta
            self.distance_raster.save(output_path)
            self.distance_raster.mask(mask_layer, output_path)

        elif self.distance == 'travel_time':
            # layer, meta = align_raster(self.path, self.friction.path, 
            # method='nearest')
            # self.friction.layer = layer
            # self.friction.meta = meta
            self.travel_time(output_path)

        else:
            self.distance_raster = self
            # .layer.copy()
            # meta = self.meta.copy()
            # meta.update(nodata=np.nan, dtype='float64')
            # self.distance_raster = RasterLayer(self.category,
            #                                    self.name + ' - log',
            #                                    distance_limit=self.distance_limit,
            #                                    inverse=self.inverse,
            #                                    normalization=self.normalization)
            # self.distance_raster.layer = layer
            # self.distance_raster.meta = meta
            # self.distance_raster.bounds = self.bounds
            # self.distance_raster.save(output_path)

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
            mask_raster(output_file, mask_layer,
                        output_file, np.nan, 'DEFLATE')
            self.normalized = RasterLayer(self.category, self.name + ' - normalized',
                                          layer_path=output_file)

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
        layer, meta = align_raster(base_layer, self.path,
                                   method=self.resample)
        self.layer = layer
        self.meta = meta
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

    def quantiles(self, quantiles):
        x = self.layer.flat
        x = x[~np.isnan(x)].copy()
        qs = np.quantile(x, quantiles)
        layer = self.layer.copy()
        i = 0
        while i < (len(qs)):
            if i == 0:
                layer[(layer >= 0) & (layer < qs[i])] = qs[i]
            else:
                layer[(layer >= qs[i - 1]) & (layer < qs[i])] = qs[i]
        return layer

    def category_legend(self, im, categories, legend_position=(1.05, 1)):
        values = list(categories.values())
        titles = list(categories.keys())

        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label="{}".format(titles[i])) for i in range(len(values))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=legend_position, loc=2, borderaxespad=0.)

        plt.grid(True)

    def plot(self, cmap='viridis', ticks=None, tick_labels=None,
             cumulative_count=None, quantiles=None, categories=None, legend_position=(1.05, 1),
             admin_layer=None):
        extent = [self.bounds[0], self.bounds[2],
                  self.bounds[1], self.bounds[3]]  # [left, right, bottom, top]

        if cumulative_count:
            layer = self.cumulative_count(cumulative_count)
        elif quantiles:
            layer = self.quantiles(quantiles)
        else:
            layer = self.layer

        layer[layer == self.meta['nodata']] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        cax = ax.imshow(layer, cmap=cmap, extent=extent)

        ax.set_axis_off()
        if categories:
            self.category_legend(cax, categories, legend_position=legend_position)
        else:
            colorbar = dict(shrink=0.8)
            if ticks:
                colorbar['ticks'] = ticks
            cbar = fig.colorbar(cax, **colorbar)

            if tick_labels:
                cbar.ax.set_yticklabels(tick_labels)
            cbar.ax.set_ylabel(self.name.replace('_', ' '))
        if isinstance(admin_layer, gpd.GeoDataFrame):
            admin_layer.plot(color='lightgrey', linewidth=1, ax=ax, zorder=0)
        plt.close()
        return fig

    def save_png(self, output_path, cmap='viridis', ticks=None, tick_labels=None,
                 cumulative_count=None, categories=None, legend_position=(1.05, 1),
                 admin_layer=None):
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path,
                                   self.name + '.png')
        fig = self.plot(cmap=cmap, ticks=ticks, tick_labels=tick_labels, cumulative_count=cumulative_count,
                        categories=categories, legend_position=legend_position,
                        admin_layer=admin_layer)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
