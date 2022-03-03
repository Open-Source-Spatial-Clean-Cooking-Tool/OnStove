import os
import geopandas as gpd
import re
import pandas as pd
import datetime

from osgeo import gdal
from rasterio import windows
from rasterio.transform import array_bounds
from rasterio import warp, features
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgb, to_hex
import time, sys

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

class Layer:
    """
    Template Layer initializing all needed variables.
    """

    def __init__(self, category='', name='', layer_path=None,
                 conn=None, normalization=None,
                 inverse=False, distance=None,
                 distance_limit=None):
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

    def travel_time(self, output_path=None, condition=None):
        layer = self.friction.layer.copy()
        layer *= (1000 / 60)  # to convert to hours per kilometer
        layer[np.isnan(layer)] = float('inf')
        layer[layer == self.friction.meta['nodata']] = float('inf')
        layer[layer < 0] = float('inf')
        mcp = MCP_Geometric(layer, fully_connected=True)
        row, col = self.start_points(condition=condition)
        pointlist = np.column_stack((row, col))
        # TODO: create method for restricted areas
        if len(pointlist)>0:
            cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
            cumulative_costs[np.where(cumulative_costs == float('inf'))] = np.nan
        else:
            cumulative_costs = np.full(self.friction.layer.shape, 2.0)

        self.distance_raster = RasterLayer(self.category,
                                           self.name + ' - traveltime',
                                           distance_limit=self.distance_limit,
                                           inverse=self.inverse,
                                           normalization=self.normalization)

        self.distance_raster.layer = cumulative_costs  # + (self.friction.layer * 1000 / 60)
        self.distance_raster.meta = self.friction.meta.copy()
        self.distance_raster.bounds = self.friction.bounds
        if output_path:
            self.distance_raster.save(output_path)

class VectorLayer(Layer):
    """
    Layer class for GIS Vector data. It stores a GeoPandas dataframe and some
    required metadata as layer name, normalization algorithm and 
    distance algorithm.
    """

    def __init__(self, category, name, layer_path=None, conn=None, query=None,
                 normalization='MinMax', inverse=False, distance='proximity',
                 distance_limit=None, bbox=None):
        """
        Initializes the class. It recibes the name of the layer, 
        the path of the layer, a normalization algorithm, a distance algorithm 
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.style = {}
        super().__init__(category=category, name=name,
                         layer_path=layer_path, conn=conn,
                         normalization=normalization, inverse=inverse,
                         distance=distance,
                         distance_limit=distance_limit)
        self.read_layer(layer_path, conn, bbox=bbox)
        if query:
            self.layer = self.layer.query(query)

    def __repr__(self):
        return 'Vector' + super().__repr__()

    def __str__(self):
        return 'Vector' + super().__str__()

    def bounds(self):
        bounds = self.layer.dissolve().bounds
        return bounds.iloc[0].to_list()

    def read_layer(self, layer_path, conn=None, bbox=None):
        if layer_path:
            if conn:
                sql = f'SELECT * FROM {layer_path}'
                self.layer = gpd.read_postgis(sql, conn)
            else:
                if isinstance(bbox, gpd.GeoDataFrame):
                    bbox = bbox.dissolve()
                else:
                    bbox = None
                self.layer = gpd.read_file(layer_path, bbox=bbox)
        self.path = layer_path

    def mask(self, mask_layer, output_path, all_touched=True):
        self.layer = gpd.clip(self.layer, mask_layer.to_crs(self.layer.crs))
        self.save(output_path)

    def reproject(self, crs, output_path):
        if self.layer.crs != crs:
            self.layer.to_crs(crs, inplace=True)
            self.save(output_path)

    def get_distance_raster(self, base_layer, output_path=None, create_raster=True):
        if self.distance == 'proximity':
            with rasterio.open(base_layer) as src:
                bounds = src.bounds
                width = src.width
                height = src.height
                crs = src.crs
                transform = src.transform

            data, meta = self.rasterize(value=1, width=width, height=height,
                                        transform=transform)

            drv = gdal.GetDriverByName('MEM')
            src_ds = drv.Create('',
                                width, height, 1,
                                gdal.GetDataTypeByName('Float32'))
            src_ds.SetGeoTransform(transform.to_gdal())
            src_ds.SetProjection(crs.wkt)
            src_ds.WriteArray(data)
            srcband = src_ds.GetRasterBand(1)

            drv = gdal.GetDriverByName('MEM')
            dst_ds = drv.Create('',
                                width, height, 1,
                                gdal.GetDataTypeByName('Float32'))

            dst_ds.SetGeoTransform(transform.to_gdal())
            dst_ds.SetProjection(crs.wkt)

            dstband = dst_ds.GetRasterBand(1)

            gdal.ComputeProximity(srcband, dstband,
                                  ["VALUES=1",
                                   "DISTUNITS=GEO"])
            data = dstband.ReadAsArray()

            meta.update(nodata=np.nan, dtype='float32')

            if create_raster:
                self.distance_raster = RasterLayer(self.category,
                                                   self.name + '_dist',
                                                   distance_limit=self.distance_limit,
                                                   inverse=self.inverse,
                                                   normalization=self.normalization)
                self.distance_raster.layer = data
                self.distance_raster.meta = meta
                self.distance_raster.bounds = bounds
                if output_path:
                    self.distance_raster.save(output_path)
            else:
                return data, meta

        elif self.distance == 'travel_time':
            self.travel_time(output_path)

    def rasterize(self, attribute=None, value=1, width=None, height=None,
                  transform=None, cell_width=None, cell_height=None,
                  output=None, nodata=0, dtype=rasterio.uint8):
        """
        Rasterizes the vector data by taking either a transform and the width
        and height of the image, or by taking the cell size and the total
        bounds of the vector layer.
        """
        if width is None:
            total_bounds = self.layer['geometry'].total_bounds
            height = round((total_bounds[3] - total_bounds[1]) / cell_width)
            width = round((total_bounds[2] - total_bounds[0]) / cell_height)

        if transform is None:
            transform = rasterio.transform.from_bounds(*self.layer['geometry'].total_bounds, width, height)

        if attribute:
            shapes = ((g, v) for v, g in zip(self.layer[attribute].values, self.layer['geometry'].values))
        else:
            shapes = ((g, value) for g in self.layer['geometry'].values)

        rasterized = features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            all_touched=True,
            dtype=rasterio.uint8)
        meta = dict(driver='GTiff',
                    dtype=dtype,
                    count=1,
                    crs=self.layer.crs,
                    width=width,
                    height=height,
                    transform=transform,
                    nodata=nodata)
        if output:
            os.makedirs(output, exist_ok=True)
            with rasterio.open(os.path.join(output, self.name + '.tif'), 'w', **meta) as dst:
                dst.write(rasterized, indexes=1)
        else:
            return rasterized, meta

    def start_points(self, condition=None):
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

    def plot(self, ax=None, style=None):
        if style is None:
            style = self.style

        if ax is None:
            self.layer.plot(**style,
                            label=self.name)
            lgnd = ax.legend(loc="upper right", prop={'size': 12})
            lgnd.legendHandles[0]._sizes = [60]
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

    def __init__(self, category, name, layer_path=None, conn=None,
                 normalization='MinMax', inverse=False, distance='proximity',
                 distance_limit=None, resample='nearest', window=None,
                 rescale=False):
        """
        Initializes the class. It recibes the name of the layer,
        the path of the layer, a normalization algorithm, a distance algorithm
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.resample = resample
        self.rescale = rescale
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

    def mask(self, mask_layer, output_path, all_touched=False):
        output_file = os.path.join(output_path,
                                   self.name + '.tif')
        mask_raster(self.path, mask_layer.to_crs(self.meta['crs']),
                    output_file, self.meta['nodata'], 'DEFLATE', all_touched=all_touched)
        self.read_layer(output_file)

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
                os.makedirs(output_path, exist_ok=True)
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
            self.travel_time(output_path)

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
            mask_raster(output_file, mask_layer,
                        output_file, np.nan, 'DEFLATE')
            self.normalized = RasterLayer(self.category, self.name + ' - normalized',
                                          layer_path=output_file)

    def polygonize(self, mask=None):
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

    def category_legend(self, im, categories, legend_position=(1.05, 1), title='', legend_cols=1):
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
             admin_layer=None, title=None, ax=None, dpi=150, legend=True, legend_title='', legend_cols=1,
             rasterized=True):
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

        #layer = layer.astype('float32')

        layer[layer == self.meta['nodata']] = np.nan

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=dpi)

        if isinstance(cmap, dict):
            values = np.sort(np.unique(layer[~np.isnan(layer)]))
            cmap = ListedColormap([to_rgb(cmap[i]) for i in values])
        cax = ax.imshow(layer, cmap=cmap, extent=extent, interpolation='none', zorder=1, rasterized=rasterized)

        if legend:
            if categories:
                self.category_legend(cax, categories, legend_position=legend_position,
                                     title=legend_title, legend_cols=legend_cols)
            else:
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

    def save_image(self, output_path, type='png', cmap='viridis', ticks=None, tick_labels=None,
                   cumulative_count=None, categories=None, legend_position=(1.05, 1),
                   admin_layer=None, title=None, ax=None, dpi=300,
                   legend=True, legend_title='', legend_cols=1, rasterized=True):
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path,
                                   self.name + f'.{type}')
        self.plot(cmap=cmap, ticks=ticks, tick_labels=tick_labels, cumulative_count=cumulative_count,
                  categories=categories, legend_position=legend_position, rasterized=rasterized,
                  admin_layer=admin_layer, title=title, ax=ax, dpi=dpi,
                  legend=legend, legend_title=legend_title, legend_cols=legend_cols)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()

    def save_style(self, output_path, cmap='magma', quantiles=False):
        if quantiles:
            qs = self.get_quantiles((0, 0.25, 0.5, 0.75, 1))
        else:
            qs = np.linspace(np.nanmin(self.layer), np.nanmax(self.layer), num=5)
        colors = plt.get_cmap(cmap, 5).colors
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
            <sld:ColorMap type="ramp">
              <sld:ColorMapEntry label="{int(qs[0])}" color="{to_hex(colors[0])}" quantity="{qs[0]}"/>
              <sld:ColorMapEntry label="{int(qs[1])}" color="{to_hex(colors[1])}" quantity="{qs[1]}"/>
              <sld:ColorMapEntry label="{int(qs[2])}" color="{to_hex(colors[2])}" quantity="{qs[2]}"/>
              <sld:ColorMapEntry label="{int(qs[3])}" color="{to_hex(colors[3])}" quantity="{qs[3]}"/>
              <sld:ColorMapEntry label="{int(qs[4])}" color="{to_hex(colors[4])}" quantity="{qs[4]}"/>
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
