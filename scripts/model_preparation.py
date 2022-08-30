import sys, os
from decouple import config

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove import VectorLayer, RasterLayer, OnStove

# 1. Create an OnSSTOVE model
output_directory = snakemake.params.output_directory
country = snakemake.params.country
model = OnStove()
model.output_directory = output_directory

# 2. Read the model data
print(f'[{country}] Read model data')
path = snakemake.input.prep_file
model.read_scenario_data(path, delimiter=',')

# 3. Add a country mask layer
path = snakemake.input.mask_layer
mask_layer = VectorLayer('admin', 'adm_1', path=path)
model.mask_layer = mask_layer

# 4. Add a population base layer
path = snakemake.input.population
model.add_layer(category='Demographics', name='Population', path=path, layer_type='raster', base_layer=True)
model.population_to_dataframe()

# 5. Calibrate population and urban/rural split
print(f'[{country}] Calibrating population')
model.calibrate_current_pop()

# path = os.path.join(output_directory, 'Demographics', 'Urban_rural_divide', 'Urban_rural_divide.tif')
ghs_path = snakemake.input.ghs
model.calibrate_urban_current_and_future_GHS(ghs_path)

# 6. Add wealth index GIS data
print(f'[{country}] Adding wealth index')
wealth_index = snakemake.params.wealth_index
if country in ['SOM', 'SDN', 'SSD']:
    model.extract_wealth_index(wealth_index + '.shp', file_type='polygon',
                               x_column="longitude", y_column="latitude", wealth_column="rwi")
else:
    model.extract_wealth_index(wealth_index + '.csv', file_type='csv',
                               x_column="longitude", y_column="latitude", wealth_column="rwi")

# 8. Read electricity network GIS layers
# Read MV lines
path = snakemake.input.mv_lines
mv_lines = VectorLayer('Electricity', 'MV_lines', path=path, distance_method='proximity')

# Read HV lines
# path = snakemake.input.hv_lines
# hv_lines = VectorLayer('Electricity', 'HV_lines', layer_path=path)

# 8.1. Calculate distance to electricity infrastructure
print(f'[{country}] Calculating distance to electricity')
model.distance_to_electricity(mv_lines=mv_lines)

# 8.2. Add night time lights data
path = snakemake.input.ntl
ntl = RasterLayer('Electricity', 'Night_time_lights', path=path)

model.raster_to_dataframe(ntl.data, name='Night_lights', method='read',
                          nodata=ntl.meta['nodata'], fill_nodata='interpolate')

# 9. Calibrate current electrified population
print(f'[{country}] Calibrating current electrified')
model.current_elec()
model.final_elec()

print(f'[{country}] Calibrated grid electrified population fraction:',
      model.gdf['Elec_pop_calib'].sum() / model.gdf['Calibrated_pop'].sum())

# 10. Read the cooking technologies data
print(f'[{country}] Reading tech data')
path = snakemake.input.techs_file
model.read_tech_data(path, delimiter=',')

# Adding tiers data to Electricity
# print(f'[{country}] Electricity tiers data')
# model.techs['Electricity'].tiers_path = snakemake.input.tiers

# 12. Reading GIS data for LPG supply
print(f'[{country}] LPG data')
travel_time = RasterLayer('LPG', 'Traveltime', snakemake.input.traveltime_cities)
model.techs['LPG'].travel_time = model.raster_to_dataframe(travel_time.data,
                                                           nodata=travel_time.meta['nodata'],
                                                           fill_nodata='interpolate', method='read') * 2 / 60

# 13. Adding GIS data for Traditional Biomass
print(f'[{country}] Traditional Biomass collected data')
model.techs['Collected_Traditional_Biomass'].friction_path = snakemake.input.biomass_friction
model.techs['Collected_Traditional_Biomass'].forest_path = snakemake.input.forest
model.techs['Collected_Traditional_Biomass'].forest_condition = lambda x: x > 0

# 14. Adding GIS data for Improved Biomass (ICS biomass)
print(f'[{country}] Improved Biomass collected data')
model.techs['Collected_Improved_Biomass'].friction_path = snakemake.input.biomass_friction
model.techs['Collected_Improved_Biomass'].forest_path = snakemake.input.forest
model.techs['Collected_Improved_Biomass'].forest_condition = lambda x: x > 0

model.techs['Biomass Forced Draft'].friction_path = snakemake.input.biomass_friction
model.techs['Biomass Forced Draft'].forest_path = snakemake.input.forest
model.techs['Biomass Forced Draft'].forest_condition = lambda x: x > 0

# 15. Adding GIS data for Improved Biomass collected (ICS biomass)
if 'Biogas' in model.techs.keys():
    # TODO: Need to finish this, add livestock data
    print(f'[{country}] Adding biogas data')
    buffaloes = snakemake.input.buffaloes
    cattles = snakemake.input.cattles
    poultry = snakemake.input.poultry
    goats = snakemake.input.goats
    pigs = snakemake.input.pigs
    sheeps = snakemake.input.sheeps
    model.techs['Biogas'].temperature = RasterLayer('Biogas', 'Temperature', snakemake.input.temperature)
    model.techs['Biogas'].water = RasterLayer('Biogas', 'Water scarcity', snakemake.input.water)

    print(f'[{country}] Recalibrating livestock')
    model.techs['Biogas'].recalibrate_livestock(model, buffaloes,
                                                cattles, poultry, goats, pigs, sheeps)
    model.techs['Biogas'].friction_path = snakemake.input.biomass_friction

# 15. Saving the prepared model inputs
print(f'[{country}] Saving the model')
model.to_pickle("model.pkl")
