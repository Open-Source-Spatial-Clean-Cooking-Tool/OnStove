import os
import geopandas as gpd
import re
from skimage.graph.mcp import MCP_Geometric

from .raster import *


class VectorLayer():
    """
    Layer class for GIS Vector data. It stores a GeoPandas dataframe and some
    required metadata as layer name, normalization algorithm and 
    distance algorithm.
    """
    friction = None
    def __init__(self, category, name, layer_path, conn=None, query=None,
                 normalization='MinMax', inverse=False, distance='proximity', 
                 distance_limit=float('inf')):
        """
        Initializes the class. It recibes the name of the layer, 
        the path of the layer, a normalization algorithm, a distance algorithm 
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.category = category
        self.name = name
        self.normalization = normalization
        self.distance = distance
        self.distance_limit = distance_limit
        self.inverse = inverse
        self.read_file(layer_path, conn)
        
        if query:
            self.layer = self.layer.query(query)
    
    def __repr__(self):
        return 'VectorLayer(name=%r)' % self.name
        
        
    def __str__(self):
        return f'VectorLayer\n    - Name: {self.name}\n' + \
               f'    - Normalization: {self.norm}'
               
    
    def read_file(self, layer_path, conn=None):
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
    
    
    def distance_raster(self, base_layer, output_path, 
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
            self.distance_raster_path = output_proximity
        elif self.distance == 'travel_time':
            self.distance_raster_path = self.path.split('.gpkg')[0]
            self.distance_raster_path = self.distance_raster_path + ' - traveltime.tif'
            
            self.friction.layer *= 1000/60 # to convert to hours per kilometer
            self.friction.layer[np.isnan(self.friction.layer)] = float('inf')
            mcp = MCP_Geometric(self.friction.layer, fully_connected=True)
            row, col = friction_start_points(self.friction.path, self.layer)
            pointlist = np.column_stack((row, col))
            cumulative_costs, traceback = mcp.find_costs(starts=pointlist)
            cumulative_costs[np.where(cumulative_costs==float('inf'))] = np.nan
            
            new_dataset = rasterio.open(self.distance_raster_path, 'w', driver='GTiff',
                            height = self.friction.layer.shape[0], width = self.friction.layer.shape[1],
                            count=1, dtype="float64",
                            crs=self.friction.meta['crs'], transform = self.friction.meta['transform'])

            new_dataset.write(cumulative_costs, 1)
            new_dataset.close()
            
    
    def normalize(self, output_path, mask_layer):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path, 
                                       self.name + ' - normalized.tif')
            normalize(self.distance_raster_path, limit=self.distance_limit, 
                      inverse=self.inverse, output_file=output_file)
            mask_raster(output_file, mask_layer, 
                        output_file, np.nan, 'DEFLATE')
    
    
    def save(self, output_path):
        output_file = os.path.join(output_path, 
                                   self.name + '.gpkg')
        os.makedirs(output_path, exist_ok=True)
        self.layer.to_file(output_file, driver="GPKG")
        self.path = output_file
        
        
    def add_friction_raster(self, raster_path):
        self.friction = RasterLayer(self.category, self.name + ' - friction', raster_path)
            
                       
class RasterLayer():
    """
    Layer class for GIS Raste data. It stores a Numpy array and the metadata 
    of a rasterio object. Also some extra metadata is stored as layer name and 
    normalization algorithm.
    """
    friction = None
    def __init__(self, category, name, layer_path,
                 normalization=None, inverse=False, 
                 distance=None, distance_limit=float('inf')):
        self.category = category
        self.name = name
        self.normalization = normalization
        self.read_layer(layer_path)
        self.distance = distance
        self.distance_limit = distance_limit
        self.inverse = inverse

    
    def __repr__(self):
        return 'RasterLayer(name=%r)' % self.name
        
        
    def __str__(self):
        return f'RasterLayer\n    - Name: {self.name}\n' + \
               f'    - Normalization: {self.norm}'
               
               
    def read_layer(self, layer_path):
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
        
    
    def reproject(self, raster_path, dst_crs, output_path, 
                  dst_width=None, dst_height=None, method='nearest'):
        if self.meta['crs'] != dst_crs:
            output_file = os.path.join(output_path, 
                                       self.name + ' - reprojected.tif')
            reproject_raster(raster_path, dst_crs, output_file=output_file, 
                             dst_width=dst_width, dst_height=dst_height, 
                             method=method, compression='DEFLATE')
                             
            self.read_layer(output_file)
        
    
    def distance_raster(self, base_layer, output_path, 
                        mask_layer):
        if self.distance == 'log':
            layer = self.layer.copy()
            layer[layer==0] = np.nan
            layer[layer>0] = np.log(layer[layer>0])
            layer = np.nan_to_num(layer, nan=0)
            layer[layer<0] = np.nan
            
            meta = self.meta.copy()
            meta.update(nodata=np.nan, dtype='float64')
            
            output_file = os.path.join(output_path, 
                                   self.name + ' - log.tif')
            
            with rasterio.open(output_file, "w", **meta) as dest:
                dest.write(layer, indexes=1)
            
            mask_raster(output_file, mask_layer, 
                        output_file, np.nan, 'DEFLATE')
                        
            self.distance_raster_path = output_file
    
    def normalize(self, output_path, mask_layer):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path, 
                                       self.name + ' - normalized.tif')
            normalize(self.distance_raster_path, limit=self.distance_limit, 
                      inverse=self.inverse, output_file=output_file)
            mask_raster(output_file, mask_layer, 
                        output_file, np.nan, 'DEFLATE')
                    
                    
    def save(self, output_path):
        output_file = os.path.join(output_path, 
                                   self.name + '.tif')
        os.makedirs(output_path, exist_ok=True)
        with rasterio.open(output_file, "w", **self.meta) as dest:
            dest.write(self.layer, indexes=1)
    
    
