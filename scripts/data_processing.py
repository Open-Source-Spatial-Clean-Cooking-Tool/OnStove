import os, sys
import geopandas as gpd
import numpy as np
import plotly.express as px

from onsstove.onsstove import OnSSTOVE
from onsstove.layer import RasterLayer, VectorLayer
from onsstove.raster import interpolate


# Create the model
output_directory = snakemake.input.output_directory
model = OnSSTOVE(project_crs=3395, cell_size=(1000, 1000))
model.output_directory = output_directory

# Add a country mask layer
country_name = snakemake.input.country_name
path = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp"
africa = gpd.read_file(path)
country = africa.loc[africa['GID_0'] == country_name.upper()]

mask_layer = VectorLayer('admin', 'adm_0')
mask_layer.layer = country
os.makedirs(f'{output_directory}/admin/adm_0', exist_ok=True)
mask_layer.reproject(model.project_crs, f'{output_directory}/admin/adm_0')
model.mask_layer = mask_layer

# Add a population base layer
path = snakemake.input.population
os.makedirs(f'{output_directory}/demographics/population', exist_ok=True)
population = RasterLayer('demographics', 'population', layer_path=path, resample='sum')
population.reproject(model.project_crs, f'{output_directory}/demographics/population',
                     model.cell_size[0], model.cell_size[1])
population.mask(model.mask_layer.layer, f'{output_directory}/demographics/population')
model.base_layer = population
model.population_to_dataframe(population)


