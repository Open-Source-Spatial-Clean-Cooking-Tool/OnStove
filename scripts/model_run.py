import sys
sys.path.append(r"C:\Users\camilorg\Box Sync\OnSSTOVE")
import os

import numpy as np

from onsstove.onsstove import OnSSTOVE

# 1. Read the OnSSTOVE model
country = snakemake.params.country
print(f'[{country}] Reading model')
model = OnSSTOVE.read_model(snakemake.input.model)

# 2. Read the scenario data
print(f'[{country}] Scenario data')
path = snakemake.input.specs_file
model.read_scenario_data(path, delimiter=',')

# 3. Calculating benefits and costs of each technology and getting the max benefit technology for each cell
model.run(technologies='all')

# 4. Printing the results
summary = model.gdf.groupby(['max_benefit_tech']).agg({'Calibrated_pop': lambda row: np.nansum(row) / 1000000,
                                             'maximum_net_benefit': lambda row: np.nansum(row) / 1000000,
                                             'deaths_avoided': 'sum',
                                             'health_costs_avoided': lambda row: np.nansum(row) / 1000000,
                                             'time_saved': 'sum',
                                             'reduced_emissions': lambda row: np.nansum(row) / 1000000000,
                                             'investment_costs': lambda row: np.nansum(row) / 1000000,
                                             'fuel_costs': lambda row: np.nansum(row) / 1000000,
                                             'emissions_costs_saved': lambda row: np.nansum(row) / 1000000})

# 5. Saving data to raster files
model.to_raster('max_benefit_tech')
model.to_raster('net_benefit_Electricity')
model.to_raster('net_benefit_LPG')
model.to_raster('net_benefit_Collected_Traditional_Biomass')
model.to_raster('net_benefit_Collected_Imporved_Biomass')
model.to_raster('maximum_net_benefit')
# model.to_raster('investment_costs')

model.to_image('maximum_net_benefit', cmap='Spectral', cumulative_count=[0.01, 0.99],
               title=f'Maximum net benefit | {country}', dpi=600)
model.to_image('max_benefit_tech', cmap='tab10', legend_position=(1, 0.9),
               title=f'Maximum benefit technology | {country}', dpi=600)

print(f'[{country}] Saving the results')
summary.to_csv(os.path.join(model.output_directory, 'Output', 'summary.csv'))
model.gdf.to_csv(os.path.join(model.output_directory, 'Output', 'results.csv'))
