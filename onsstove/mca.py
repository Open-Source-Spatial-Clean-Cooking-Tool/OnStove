import os
import psycopg2
import pandas as pd
import geopandas as gpd
import numpy as np

import onsstove.technology
from .raster import *
from .layer import VectorLayer, RasterLayer


class OnSSTOVE():
    """
    Class containing the methods to perform a Multi Criteria Analysis
    of clean cooking access potential. It calculates a Clean Cooking Demand 
    Index, a Clean Cooking Supply Index and a Clean Cooking Potential Index
    """
    conn = None
    base_layer = None

    population = 'path_to_file'

    
    def __init__(self, project_crs=None, cell_size=None, output_directory='output'):
        """
        Initializes the class and sets an empty layers dictionaries.
        """
        self.layers = {}
        self.project_crs = project_crs
        self.cell_size = cell_size
        self.output_directory = output_directory
        self.mask_layer = None
        
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
        output_path = os.path.join(self.output_directory, category, name)
        os.makedirs(output_path, exist_ok=True)
        
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
            layer = RasterLayer(category, name, layer_path, 
                                normalization=normalization, inverse=inverse,
                                distance=distance, resample=resample)
            
            if base_layer:
                if not self.cell_size:
                    self.cell_size = (layer.meta['transform'][0], 
                                      abs(layer.meta['transform'][1]))
                if not self.project_crs:
                    self.project_crs = layer.meta['crs']
                    
                cell_size_diff = abs(self.cell_size[0] - layer.meta['transform'][0]) / \
                                 layer.meta['transform'][0]
                                 
                if (layer.meta['crs'] != self.project_crs) or (cell_size_diff>0.01):
                    layer.reproject(self.project_crs, 
                                    output_path=output_path, 
                                    cell_width=self.cell_size[0], 
                                    cell_height=self.cell_size[1], 
                                    method=resample)
                    
                self.base_layer = layer

                
        if category in self.layers.keys():
            self.layers[category][name] = layer
        else:
            self.layers[category] = {name: layer}
            
    
    def add_mask_layer(self, name, layer_path, postgres=False):
        """
        Adds a vector layer to self.mask_layer, which will be used to mask all 
        other layers into is boundaries
        """
        if postgres:
            sql = f'SELECT * FROM {layer_path}'
            self.mask_layer = gpd.read_postgis(sql, self.conn)
        else:
            self.mask_layer = gpd.read_file(layer_path)
            
        if self.mask_layer.crs != self.project_crs:
            self.mask_layer.to_crs(self.project_crs, inplace=True)
            
     
    def mask_layers(self, datasets='all'):
        """
        Uses the previously added mask layer in self.mask_layer to mask all 
        other layers to its boundaries
        """
        if not isinstance(self.mask_layer, gpd.GeoDataFrame):
            raise Exception('The `mask_layer` attribute is empty, please first ' + \
            'add a mask layer using the `.add_mask_layer` method.')
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory, 
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.mask(self.mask_layer, output_path) 
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.mask(self.mask_layer, output_path) 
    

    def align_layers(self, datasets='all'):
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory,
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                if isinstance(layer, VectorLayer):
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(self.base_layer.path, output_path)
                else: 
                    if name != self.base_layer.name:
                        layer.align(self.base_layer.path, output_path)
                    if isinstance(layer.friction, RasterLayer):
                        layer.friction.align(self.base_layer.path, output_path)
  
    def reproject_layers(self, datasets='all'):
        """
        Goes through all layer and call their `.reproject` method with the 
        `project_crs` as argument
        """
        datasets = self.get_layers(datasets)
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
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory, 
                                           category, name)
                os.makedirs(output_path, exist_ok=True)
                layer.get_distance_raster(self.base_layer.path, 
                                          output_path, self.mask_layer)
                if isinstance(layer.friction, RasterLayer):
                    layer.friction.get_distance_raster(self.base_layer.path, 
                                                   output_path, self.mask_layer)
            
    
    def normalize_rasters(self, datasets='all'):
        """
        Goes through all layer and call their `.normalize` method
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory, 
                                           category, name)
                layer.normalize(output_path, self.mask_layer)

    
    def save_datasets(self, datasets='all'):
        """
        Saves all layers that have not been previously saved
        """
        datasets = self.get_layers(datasets)
        for category, layers in datasets.items():
            for name, layer in layers.items():
                output_path = os.path.join(self.output_directory, 
                                           category, name)
                layer.save(output_path)
        
        
        
        
        
        