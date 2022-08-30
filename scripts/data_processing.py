import sys
from decouple import config
import os
onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)
import numpy as np

from onstove import DataProcessor, VectorLayer

# 1. Create a data processor
output_directory = snakemake.params.output_directory
country = snakemake.params.country
data = DataProcessor(project_crs=3857, cell_size=(1000, 1000))
data.output_directory = output_directory

# 2. Add a mask layer (country boundaries)
print(f'[{country}] Adding mask layer')
adm_path = snakemake.input.mask_layer
data.add_mask_layer(category='Administrative', name='Country_boundaries',
                    path=adm_path, query=f"GID_0 == '{country}'")

# 3. Add GIS layers

# Demographics
print(f'[{country}] Adding population')
pop_path = snakemake.input.population
data.add_layer(category='Demographics', name='Population', path=pop_path,
               layer_type='raster', base_layer=True, resample='sum')

ghs_path = snakemake.input.ghs
data.add_layer(category='Demographics', name='Urban', path=ghs_path, layer_type='raster',
               resample='nearest')

# Biomass
print(f'[{country}] Adding forest')
forest_path = snakemake.input.forest
data.add_layer(category='Biomass', name='Forest', path=forest_path, layer_type='raster',
               resample='sum')
data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data < 5] = 0
data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data >= 5] = 1
data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
transform = data.layers['Biomass']['Forest'].calculate_default_transform(data.project_crs)[0]
factor = (data.cell_size[0] ** 2) / (transform[0] ** 2)

print(f'[{country}] Adding walking friction')
friction_path = snakemake.input.walking_friction
data.add_layer(category='Biomass', name='Friction', path=friction_path, layer_type='raster',
               resample='average', window=True)

# Electricity
print(f'[{country}] Adding MV lines')
mv_path = snakemake.input.mv_lines
data.add_layer(category='Electricity', name='MV_lines', path=mv_path,
               layer_type='vector', window=True)

print(f'[{country}] Adding Nighttime Lights')
ntl_path = snakemake.input.ntl
data.add_layer(category='Electricity', name='Night_time_lights', path=ntl_path, layer_type='raster',
               resample='average', window=True)
data.layers['Electricity']['Night_time_lights'].save(f'{data.output_directory}/Electricity/Night_time_lights')

# LPG
print(f'[{country}] Adding traveltime to cities')
traveltime_cities = snakemake.input.traveltime_cities
data.add_layer(category='LPG', name='Traveltime', path=traveltime_cities,
               layer_type='raster', resample='average', window=True)
data.layers['LPG']['Traveltime'].save(f'{data.output_directory}/LPG/Traveltime')

# Temperature
print(f'[{country}] Adding temperature')
temperature = snakemake.input.temperature
data.add_layer(category='Biogas', name='Temperature', path=temperature,
               layer_type='raster', resample='average', window=True)
data.layers['Biogas']['Temperature'].save(f'{data.output_directory}/Biogas/Temperature')
data.mask_layers(datasets={'Biogas': ['Temperature']})

# Livestock
print(f'[{country}] Adding livestock')
buffaloes = snakemake.input.buffaloes
cattles = snakemake.input.cattles
poultry = snakemake.input.poultry
goats = snakemake.input.goats
pigs = snakemake.input.pigs
sheeps = snakemake.input.sheeps

for key, path in {'buffaloes': buffaloes,
             'cattles': cattles,
             'poultry': poultry,
             'goats': goats,
             'pigs': pigs,
             'sheeps': sheeps}.items():
    data.add_layer(category='Biogas/Livestock', name=key, path=path,
                   layer_type='raster', resample='nearest', window=True, rescale=True)

print(f'[{country}] Adding water scarcity')
water = VectorLayer('Biogas', 'Water scarcity', snakemake.input.water, bbox=data.mask_layer.data)
water.data["class"] = 0
water.data['class'] = np.where(water.data['bws_label'].isin(['Low (<10%)',
                                                               'Low - Medium (10-20%)']), 1, 0)
water.data.to_crs(data.project_crs, inplace=True)
out_folder = os.path.join(data.output_directory, "Biogas", "Water scarcity")
water.rasterize(cell_height=data.cell_size[0], cell_width=data.cell_size[1],
                attribute="class", output=out_folder, nodata=0)
data.add_layer(category='Biogas', name='Water scarcity',
               path=os.path.join(out_folder, 'Water scarcity.tif'),
               layer_type='raster', resample='nearest')

# 4. Mask reproject and align all required layers
print(f'[{country}] Aligning all layers')
data.align_layers(datasets='all')

print(f'[{country}] Masking all layers')
data.mask_layers(datasets='all')

print(f'[{country}] Reprojecting all layers')
data.reproject_layers(datasets={'Electricity': ['MV_lines']})

# Canopy calculation
print(f'[{country}] Calculating forest canopy cover')
data.layers['Biomass']['Forest'].data /= factor
data.layers['Biomass']['Forest'].data *= 100
data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data > 100] = 100
data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
