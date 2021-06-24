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
        self.layers = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        
    def get_layers(self, layers):
        if layers=='all':
            layers = self.layers
        else: 
            layers = {category: {name: self.layers[category][name]} for category, names in layers.items() for name in names}
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
                                     
    
    def add_layer(self, category, name, layer_path, layer_type, query=None,
                  postgres=False, base_layer=False, resample='nearest', 
                  normalization=None, inverse=False, distance=None, 
                  distance_limit=float('inf')):
        """
        Adds a new layer (type VectorLayer or RasterLayer) to the MCA class

        Parameters
        ----------
        arg1 : 
        """
        if layer_type=='vector':
            if postgres:
                layer = VectorLayer(category, name, layer_path, conn=self.conn,
                                    normalization=normalization, 
                                    distance=distance, 
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)
            else:
                layer = VectorLayer(category, name, layer_path, 
                                    normalization=normalization, 
                                    distance=distance,
                                    distance_limit=distance_limit,
                                    inverse=inverse, query=query)
            
        elif layer_type=='raster':
            output_path = os.path.join('output', category, name)
            layer = RasterLayer(category, name, layer_path, 
                                normalization=normalization, inverse=inverse,
                                distance=distance)
            
            if base_layer:
                self.base_layer_path = layer_path
                cell_size_diff = abs(self.cell_size[0] - layer.meta['transform'][0]) / \
                                 layer.meta['transform'][0]
                                 
                if (layer.meta['crs'] != self.project_crs) or (cell_size_diff>0.01):
                    output_path = os.path.join('output', category, name)
                    os.makedirs(output_path, exist_ok=True)
                    layer.reproject(layer_path, self.project_crs, 
                                    output_path=output_path, 
                                    dst_width=self.cell_size[0], 
                                    dst_height=self.cell_size[1], 
                                    method=resample)
                    self.base_layer_path = os.path.join(output_path, 
                                                        name + '.tif')
                    
                self.base_layer = layer

                
        if category in self.layers.keys():
            self.layers[category][name] = layer
        else:
            self.layers[category] = {name: layer}
            
    
    def add_mask_layer(self, name, layer_path, postgres=False):
        if postgres:
            sql = f'SELECT * FROM {layer_path}'
            self.mask_layer = gpd.read_postgis(sql, self.conn)
        else:
            self.mask_layer = gpd.read_file(layer_path)
            
        if self.mask_layer.crs != self.project_crs:
            self.mask_layer.to_crs(self.project_crs, inplace=True)
            
     
    def mask_layers(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join('output', category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.mask(self.mask_layer, output_path) 
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.mask(self.mask_layer, output_path) 
    
    
    def reproject_layers(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join('output', category, name)
                os.makedirs(output_path, exist_ok=True)
                if isinstance(layer, VectorLayer):
                    layer.reproject(self.project_crs, output_path)
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.reproject(layer.friction.path, 
                                                 self.project_crs, output_path)
                elif isinstance(layer, RasterLayer):
                    layer.reproject(layer.path, self.project_crs, output_path)
            
            
    def get_distance_rasters(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join('output', category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.distance_raster(self.base_layer_path, 
                                      output_path, self.mask_layer)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.distance_raster(self.base_layer_path, 
                                                   output_path, self.mask_layer)
            
    
    def normalize_rasters(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join('output', category, name)
                layer.normalize(output_path, self.mask_layer)

    
    def save_datasets(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join('output', category, name)
                layer.save(output_path)
        
        
        
        
        
        