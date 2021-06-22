import os
import geopandas as gpd

from .raster import *


class VectorLayer():
    """
    Layer class for GIS Vector data. It stores a GeoPandas dataframe and some
    required metadata as layer name, normalization algorithm and 
    distance algorithm.
    """
    def __init__(self, name, layer_path, conn=None, normalization='MinMax', 
                 distance='proximity', distance_limit=float('inf')):
        """
        Initializes the class. It recibes the name of the layer, 
        the path of the layer, a normalizaton algorithm, a distance algorithm 
        and a PostgreSQL connection if the layer needs to be read from a database
        """
        self.name = name
        self.normalization = normalization
        self.distance = distance
        self.distance_limit = distance_limit
        if conn:
            sql = f'SELECT * FROM {layer_path}'
            self.layer = gpd.read_postgis(sql, conn)
        else:
            self.layer = gpd.read_file(layer_path)
    
    
    def __repr__(self):
        return 'VectorLayer(name=%r)' % self.name
        
        
    def __str__(self):
        return f'VectorLayer\n    - Name: {self.name}\n' + \
               f'    - Normalization: {self.norm}'
               
    
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
                        
    
    def normalize(self, output_path):
        if self.normalization == 'MinMax':
            output_file = os.path.join(output_path, 
                                       self.name + ' - normalized.tif')
            normalize(self.distance_raster_path, limit=self.distance_limit, 
                      output_file=output_file)
            
                       
class RasterLayer():
    """
    Layer class for GIS Raste data. It stores a Numpy array and the metadata 
    of a rasterio object. Also some extra metadata is stored as layer name and 
    normalization algorithm.
    """
    def __init__(self, name, layer_path,
                 normalization='log'):
        self.name = name
        self.normalization = normalization
        self.read_layer(layer_path)
    
    
    def __repr__(self):
        return 'RasterLayer(name=%r)' % self.name
        
        
    def __str__(self):
        return f'RasterLayer\n    - Name: {self.name}\n' + \
               f'    - Normalization: {self.norm}'
               
               
    def read_layer(self, layer_path):
        with rasterio.open(layer_path) as src:
            self.layer = src.read(1)
            self.meta = src.meta
               
    
    def reproject(self, raster_path, dst_crs, output_file, 
                  dst_width, dst_height, method):
        reproject_raster(raster_path, dst_crs, output_file=output_file, 
                         dst_width=dst_width, dst_height=dst_height, 
                         method=method, compression='DEFLATE')
                         
        self.read_layer(output_file)
        
    
    def normalize(self, output_path):
        if self.normalization == 'log':
            layer = self.layer.copy()
            layer[layer==0] = np.nan
            layer[layer>0] = np.log(layer[layer>0])
            layer = np.nan_to_num(layer, nan=0)
            layer[layer<0] = np.nan
            layer = layer / (np.nanmax(layer) - np.nanmin(layer))
            
            meta = self.meta.copy()
            meta.update(nodata=np.nan, dtype='float64')
        else:
            layer = self.layer
            meta = self.meta
        
        output_file = os.path.join(output_path, 
                                   self.name + ' - normalized.tif')
        with rasterio.open(output_file, "w", **meta) as dest:
            dest.write(layer, indexes=1)
    
    
