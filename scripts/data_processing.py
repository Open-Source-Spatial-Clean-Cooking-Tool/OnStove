import sys
sys.path.append(r"C:\Users\khavari\Box\OnSTOVE")
from onsstove.onsstove import DataProcessor

# 1. Create a data processor
output_directory = snakemake.params.output_directory
country = snakemake.params.country
data = DataProcessor(project_crs=3857, cell_size=(1000, 1000))
data.output_directory = output_directory

# 2. Add a mask layer (country boundaries)
print(f'[{country}] Adding mask layer')
adm_path = snakemake.input.mask_layer
data.add_mask_layer(category='Administrative', name='Country_boundaries',
                    layer_path=adm_path, query=f"GID_0 == '{country}'")

# 3. Add GIS layers

# Demographics
print(f'[{country}] Adding population')
pop_path = snakemake.input.population
data.add_layer(category='Demographics', name='Population', layer_path=pop_path,
               layer_type='raster', base_layer=True, resample='sum')

ghs_path = snakemake.input.ghs
data.add_layer(category='Demographics', name='Urban', layer_path=ghs_path, layer_type='raster',
               resample='nearest')

# Biomass
print(f'[{country}] Adding forest')
forest_path = snakemake.input.forest
data.add_layer(category='Biomass', name='Forest', layer_path=forest_path, layer_type='raster',
               resample='sum')
data.layers['Biomass']['Forest'].layer[data.layers['Biomass']['Forest'].layer < 5] = 0
data.layers['Biomass']['Forest'].layer[data.layers['Biomass']['Forest'].layer >= 5] = 1
data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
transform = data.layers['Biomass']['Forest'].calculate_default_transform(data.project_crs)[0]
factor = (data.cell_size[0] ** 2) / (transform[0] ** 2)

print(f'[{country}] Adding walking friction')
friction_path = snakemake.input.walking_friction
data.add_layer(category='Biomass', name='Friction', layer_path=friction_path, layer_type='raster',
               resample='average', window=True)

# Electricity
# print('Adding HV lines')
# hv_path = snakemake.input.hv_lines
# data.add_layer(category='Electricity', name='HV_lines', layer_path=hv_path, layer_type='vector')

print(f'[{country}] Adding MV lines')
mv_path = snakemake.input.mv_lines
data.add_layer(category='Electricity', name='MV_lines', layer_path=mv_path,
               layer_type='vector', window=True)

print(f'[{country}] Adding Nighttime Lights')
ntl_path = snakemake.input.ntl
data.add_layer(category='Electricity', name='Night_time_lights', layer_path=ntl_path, layer_type='raster',
               resample='average', window=True)
data.layers['Electricity']['Night_time_lights'].save(f'{data.output_directory}/Electricity/Night_time_lights')

# LPG
print(f'[{country}] Adding traveltime to cities')
traveltime_cities = snakemake.input.traveltime_cities
data.add_layer(category='LPG', name='Traveltime', layer_path=traveltime_cities,
               layer_type='raster', resample='average', window=True)
data.layers['LPG']['Traveltime'].save(f'{data.output_directory}/LPG/Traveltime')

# Temperature
print(f'[{country}] Adding temperature')
temperature = snakemake.input.temperature
data.add_layer(category='Biogas', name='Temperature', layer_path=temperature,
               layer_type='raster', resample='average', window=True)
data.layers['Biogas']['Temperature'].save(f'{data.output_directory}/Biogas/Temperature')
data.mask_layers(datasets={'Biogas': ['Temperature']})

# 4. Mask reproject and align all required layers
# print('Reprojecting all layers')
# data.reproject_layers(datasets='all')

print(f'[{country}] Aligning all layers')
data.align_layers(datasets='all')

print(f'[{country}] Masking all layers')
data.mask_layers(datasets='all')

print(f'[{country}] Reprojecting all layers')
data.reproject_layers(datasets={'Electricity': ['MV_lines']})

# Canopy calculation
print(f'[{country}] Calculating forest canopy cover')
data.layers['Biomass']['Forest'].layer /= factor
data.layers['Biomass']['Forest'].layer *= 100
data.layers['Biomass']['Forest'].layer[data.layers['Biomass']['Forest'].layer > 100] = 100
data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
