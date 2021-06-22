import os
import psycopg2
import pandas as pd
import geopandas as gpd
import numpy as np

from .raster import *
from .layer import VectorLayer, RasterLayer


class MCA():
    """
    Class containing the methods to perform a Multi Criteria Analysis
    of clean cooking access potential. It calculates a Clean Cooking Demand 
    Index, a Clean Cooking Supply Index and a Clean Cooking Potential Index
    """
    conn = None
    base_layer = None
    
    def __init__(self, project_crs=3857, cell_size=(1000, 1000)):
        """
        Initializes the class and sets empty demand and supply lists.
        """
        self.demand = {}
        self.supply = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        
    
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
                                     
    
    def add_layer(self, category, name, layer_path, layer_type, postgres=False,
                  base_layer=False, resample='nearest', normalization='MinMax', 
                  distance=None, distance_limit=float('inf')):
        """
        Adds a new layer (type VectorLayer or RasterLayer) to the MCA class

        Parameters
        ----------
        arg1 : 
        """
        if layer_type=='vector':
            if postgres:
                layer = VectorLayer(name, layer_path, conn=self.conn,
                                    normalization=normalization, 
                                    distance=distance)
            else:
                layer = VectorLayer(name, layer_path, 
                                    normalization=normalization, 
                                    distance=distance)
                
            if layer.layer.crs != self.project_crs:
                layer.layer.to_crs(self.project_crs, inplace=True)
            
        elif layer_type=='raster':
            layer = RasterLayer(name, layer_path, normalization)
            
            if base_layer:
                self.base_layer_path = layer_path
                cell_size_diff = abs(self.cell_size[0] - layer.meta['transform'][0]) / \
                                 layer.meta['transform'][0]
                                 
                if (layer.meta['crs'] != self.project_crs) or (cell_size_diff>0.01):
                    output_file = os.path.join('output', category, name)
                    os.makedirs(output_file, exist_ok=True)
                    layer.reproject(layer_path, self.project_crs, 
                                    output_file=os.path.join(output_file, 
                                                             name + '.tif'), 
                                    dst_width=self.cell_size[0], 
                                    dst_height=self.cell_size[1], 
                                    method=resample)
                    self.base_layer_path = os.path.join(output_file, 
                                                        name + '.tif')
                    
                self.base_layer = layer
                
                
        if category=='demand':
            self.demand[name] = layer
        elif category=='supply':
            self.supply[name] = layer
            
    
    def add_mask_layer(self, name, layer_path, postgres=False):
        if postgres:
            sql = f'SELECT * FROM {layer_path}'
            self.mask_layer = gpd.read_postgis(sql, self.conn)
        else:
            self.mask_layer = gpd.read_file(layer_path)
            
        if self.mask_layer.crs != self.project_crs:
            self.mask_layer.to_crs(self.project_crs, inplace=True)
            
            
    def get_distance_rasters(self, layers='all'):
        if layers=='all':
            for name, layer in self.demand.items():
                if isinstance(layer, VectorLayer):
                    output_path = os.path.join('output', 'demand', name)
                    os.makedirs(output_path, exist_ok=True)
                    layer.distance_raster(self.base_layer_path, 
                                          output_path, self.mask_layer)
            
    
    def normalize_rasters(self, layers='all'):
        if layers=='all':
            for name, layer in self.demand.items():
                output_path = os.path.join('output', 'demand', name)
                layer.normalize(output_path)
    
    
    def produce_output(self):
        for name, layer in self.demand.items():
            rasterize(layer['layer'].to_crs(3857), 'data/population_npl_2018-10-01_1km.tif', 
                      outpul_file='rasterized.tif', nodata=0,
                      compression='DEFLATE', save=True)
        
        
        
        
        
        