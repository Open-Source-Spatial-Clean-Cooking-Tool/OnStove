import os
import geopandas as gpd
import re
import pandas as pd
import datetime
from skimage.graph.mcp import MCP_Geometric

from .raster import *


class Layer():
    """
    Template Layer initializing all needed variables.
    """
    def __init__(self, category, name, layer_path=None,
                 conn=None, normalization=None, 
                 inverse=False, distance=None, 
                 distance_limit=float('inf'), resample=None):
        self.category = category
        self.name = name
        self.normalization = normalization
        self.distance = distance
        self.distance_limit = distance_limit
        self.inverse = inverse
        self.friction = None
        self.distance_raster = None
        self.restrictions = []
        self.resample = resample
        self.read_layer(layer_path, conn)
        
    
    def __repr__(self):
        return 'Layer(name=%r)' % self.name
        
        
    def __str__(self):
        return f'Layer\n    - Name: {self.name}\n' + \
               f'    - Category: {self.category}\n' + \
               f'    - Normalization: {self.normalization}\n' + \
               f'    - Distance method: {self.distance}\n'+ \
               f'    - Distance limit: {self.distance_limit}\n' + \
               f'    - Inverse: {self.inverse}\n' + \
               f'    - Resample: {self.resample}\n' + \
               f'    - Path: {self.path}'
               
               
    def read_layer(self, layer_path, conn=None):
        pass
        
    
    def travel_time(self, output_path):
        self.friction.layer *= 1000/60 # to convert to hours per kilometer
        self.friction.layer[np.isnan(self.friction.layer)] = float('inf')
        mcp = MCP_Geometric(self.friction.layer, fully_connected=True)
        row, col = self.start_points()
        pointlist = np.column_stack((row, col))
        # TODO: create method for restricted areas
        cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
        cumulative_costs[np.where(cumulative_costs==float('inf'))] = np.nan
        
        self.distance_raster = RasterLayer(self.category, 
                                           self.name + ' - traveltime')
                                           
        self.distance_raster.layer = cumulative_costs
        self.distance_raster.meta = self.friction.meta.copy()
        self.distance_raster.save(output_path)


class VectorLayer(Layer):
    """
    Layer class for GIS Vector data. It stores a GeoPandas dataframe and some
    required metadata as layer name, normalization algorithm and 
    distance algorithm.
    """
    def __init__(self, category, name, layer_path, conn=None, query=None,
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
        
        if query:
            self.layer = self.layer.query(query)
            
            
    def __repr__(self):
        return 'Vector' + super().__repr__()
        
        
    def __str__(self):
        return 'Vector' + super().__str__()
               
    
    def read_layer(self, layer_path, conn=None):
        if conn:
            sql = f'SELECT * FROM {layer_path}'
            self.layer = gpd.read_postgis(sql, conn)
        else:
            self.layer = gpd.read_file(layer_path)
        self.path = layer_path
    
    
    def mask(self, mask_layer, output_path):
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
                                            self.name + ' - proximity.tif')
            mask_raster(output_proximity_temp, mask_layer, 
                        output_proximity, np.nan, 'DEFLATE')
            os.remove(output_rasterized)
            os.remove(output_proximity_temp)
            self.distance_raster = RasterLayer(self.category, 
                                               self.name + ' - proximity', 
                                               output_proximity)
            
        elif self.distance == 'travel_time':
            self.travel_time(output_path)
            
            
    def start_points(self):
        return friction_start_points(self.friction.path, 
                                     self.layer)
            
    
    def normalize(self, output_path, mask_layer):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path, 
                                       self.name + ' - normalized.tif')
            normalize(self.distance_raster.path, limit=self.distance_limit, 
                      inverse=self.inverse, output_file=output_file)
            mask_raster(output_file, mask_layer, 
                        output_file, np.nan, 'DEFLATE')
    
    
    def save(self, output_path):
        for column in self.layer.columns:
            if isinstance(self.layer[column].iloc[0], datetime.date):
                self.layer[column] = self.layer[column].astype('datetime64')
        output_file = os.path.join(output_path, 
                                   self.name + '.geojson')
        os.makedirs(output_path, exist_ok=True)
        self.layer.to_file(output_file, driver='GeoJSON')
        self.path = output_file
        
        
    def add_friction_raster(self, raster_path, resample='nearest'):
        self.friction = RasterLayer(self.category, self.name + ' - friction', 
                                    raster_path, resample=resample)
        
       
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
    def __repr__(self):
        return 'Raster' + super().__repr__()
        
        
    def __str__(self):
        return 'Raster' + super().__str__()
               
               
    def read_layer(self, layer_path, conn=None):
        if layer_path:
            with rasterio.open(layer_path) as src:
                self.layer = src.read(1)
                self.meta = src.meta
        self.path = layer_path
            
            
    def mask(self, mask_layer, output_path):
        output_file = os.path.join(output_path, 
                                   self.name + '.tif')
        mask_raster(self.path, mask_layer.to_crs(self.meta['crs']), 
                    output_file, np.nan, 'DEFLATE')
        self.read_layer(output_file)
        
    
    def reproject(self, crs, output_path, 
                  cell_width=None, cell_height=None, method='nearest'):
        if self.meta['crs'] != crs:
            output_file = os.path.join(output_path, 
                                       self.name + ' - reprojected.tif')
            reproject_raster(self.path, crs, output_file=output_file, 
                             cell_width=cell_width, cell_height=cell_height, 
                             method=method, compression='DEFLATE')
                             
            self.read_layer(output_file)
        
    
    def get_distance_raster(self, base_layer, output_path, 
                        mask_layer):
        if self.distance == 'log':
            layer = self.layer.copy()
            layer[layer==0] = np.nan
            layer[layer>0] = np.log(layer[layer>0])
            layer = np.nan_to_num(layer, nan=0)
            layer[layer<0] = np.nan
            
            meta = self.meta.copy()
            meta.update(nodata=np.nan, dtype='float64')
                                  
            self.distance_raster = RasterLayer(self.category,
                                               self.name + ' - log')
                                               
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
            
            
    def start_points(self):
        return np.where(np.isin(self.layer, self.starting_cells))
            
    
    def normalize(self, output_path, mask_layer):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path, 
                                       self.name + ' - normalized.tif')
            normalize(self.distance_raster.path, limit=self.distance_limit, 
                      inverse=self.inverse, output_file=output_file)
            mask_raster(output_file, mask_layer, 
                        output_file, np.nan, 'DEFLATE')
                    
                    
    def save(self, output_path, sufix=''):
        output_file = os.path.join(output_path, 
                                   self.name + f'{sufix}.tif')
        self.path = output_file
        os.makedirs(output_path, exist_ok=True)
        self.meta.update(dtype=self.layer.dtype)
        with rasterio.open(output_file, "w", **self.meta) as dest:
            dest.write(self.layer, indexes=1)
            
            
    def add_friction_raster(self, raster_path, starting_cells=[1], 
                            resample='nearest'):
        self.starting_cells = starting_cells
        self.friction = RasterLayer(self.category, 
                                    self.name + ' - friction', 
                                    raster_path, resample=resample)
                                    
                                    
    def align(self, base_layer, output_path):
        layer, meta = align_raster(base_layer, self.path, 
                                   method=self.resample)
        self.layer = layer
        self.meta = meta
        self.save(output_path, ' - aligned')
    
    
