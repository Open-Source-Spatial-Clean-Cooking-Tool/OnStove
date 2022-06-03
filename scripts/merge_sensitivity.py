import sys, os
import geopandas as gpd
import pandas as pd
from decouple import config

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove.layer import VectorLayer
from onstove.onstove import OnStove
from sensitivity import run_model

df = pd.DataFrame({'Households': [], 'max_benefit_tech': [], 'Calibrated_pop': [],
                   'maximum_net_benefit': [], 'deaths_avoided': [], 'health_costs_avoided': [],
                   'time_saved': [], 'reduced_emissions': [], 'investment_costs': [],
                   'om_costs': [], 'fuel_costs': [], 'emissions_costs_saved': [],
                   'opportunity_cost_gained': [], 'salvage_value': []})

print('Creating Africa model...')
africa = OnStove()
output_directory = os.path.join(snakemake.params.output_directory, 'Africa')
africa.output_directory = output_directory

africa.gdf = df

file_name = snakemake.params.file_name

for model_file, country, sensitivity_file in zip(snakemake.input.models, snakemake.params.countries,
                                                 snakemake.input.sensitivity_files):
    country_directory = os.path.join(snakemake.params.output_directory, country)
    model = run_model(country, model_file, sensitivity_file, snakemake.input.technology_file,
                      country_directory, file_name)
    model.gdf['country'] = country
    africa.gdf = africa.gdf.append(model.gdf[df.columns], ignore_index=True)

print('Saving results...')
os.makedirs(output_directory, exist_ok=True)
africa.summary().to_csv(snakemake.output.results, index=False)
