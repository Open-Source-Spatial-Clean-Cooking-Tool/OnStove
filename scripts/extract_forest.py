import sys, os
sys.path.append(r"C:\Users\camilorg\Box Sync\OnSSTOVE")
import rasterio
from onsstove.raster import merge_rasters
from onsstove.layer import VectorLayer
from onsstove.layer import RasterLayer

country = snakemake.params.country

if country in ['GIN', 'CIV', 'BFA', 'GHA', 'TGO', 'BEN', 'NGA', 'CMR', 'TCD', 'CAF', 'SDN', 'SSD', 'ETH', 'SOM']:
    locations = ['S', 'N']
elif country in ['GMB', 'GNB', 'GNQ', 'SEN', 'MLI', 'MRT', 'NER', 'ERI', 'DJI']:
    locations = ['N']
else:
    locations = ['S']

adm_path = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp"
adm_0 = VectorLayer('', 'Boundaries', adm_path, query=f"GID_0 == '{country}'")

bounds = adm_0.layer.dissolve().bounds
bounds = bounds.iloc[0].to_list()

out_folder = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest"

for location in locations:
    print('    ' + location)
    forest_path = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest\Forest_height_2019_{}AFR.tif".format(
        location)
    with rasterio.open(forest_path) as src:
        rast_bounds = src.bounds

    if location == 'S':
        new_bounds = [max(bounds[0], rast_bounds[0]), max(bounds[1], rast_bounds[1]),
                      min(bounds[2], rast_bounds[2]), min(bounds[3], rast_bounds[3])]
    else:
        new_bounds = bounds

    print('       - Reading raster')
    if len(locations) > 1:
        name = f'Forest_{location}'
    else:
        name = 'Forest'
    forest = RasterLayer('', name, forest_path, window=new_bounds)
    forest.layer[forest.layer > 60] = 0

    forest.layer = forest.layer.astype('byte')
    forest.meta.update(dtype='int8', nodata=0)

    print('       - Saving raster')
    forest.save(os.path.join(out_folder, country))

if len(locations) > 1:
    print('   Merging rasters')
    merge_rasters(os.path.join(out_folder, country, '*.tif'), 4326, os.path.join(out_folder, country, 'Forest.tif'))
