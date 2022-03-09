import sys, os
import geopandas as gpd
import pandas as pd
from decouple import config

sys.path.append(config('ONSSTOVE'))

from onsstove.layer import VectorLayer
from onsstove.onsstove import OnSSTOVE
from sensitivity import run_model

df = pd.DataFrame({'country': [], 'Households': [], 'max_benefit_tech': [], 'Calibrated_pop': [],
                   'maximum_net_benefit': [], 'deaths_avoided': [], 'health_costs_avoided': [],
                   'time_saved': [], 'reduced_emissions': [], 'investment_costs': [],
                   'om_costs': [], 'fuel_costs': [], 'emissions_costs_saved': [],
                   'opportunity_cost_gained': [], 'salvage_value': [], 'geometry': []})

print('Creating Africa model...')
africa = OnSSTOVE()
output_directory = os.path.join(snakemake.params.output_directory, 'Africa', snakemake.params.file_name)
africa.output_directory = output_directory

mask_layer = VectorLayer('admin', 'adm_1', layer_path=snakemake.input.boundaries)
mask_layer.layer = mask_layer.layer.to_crs(3857)
africa.mask_layer = mask_layer
africa.gdf = gpd.GeoDataFrame(df, crs='epsg:3857')

file_name = snakemake.params.file_name

for model_file, country, sensitivity_file in zip(snakemake.input.models, snakemake.params.countries,
                                                 snakemake.input.sensitivity_files):
    country_directory = os.path.join(snakemake.params.output_directory, country)
    model = run_model(country, model_file, sensitivity_file, country_directory, file_name)
    model.gdf['country'] = country
    africa.gdf = africa.gdf.append(model.gdf[df.columns], ignore_index=True)

cmap = {"Biomass ICS": '#6F4070', "LPG": '#66C5CC', "Biomass": '#FFB6C1',
        "Charcoal": '#364135', "Charcoal ICS": '#d4bdc5',
        "Biogas": '#73AF48', "Biogas and Biomass ICS": "#F6029E",
        "Biogas and LPG": "#f97b72", "Biogas and Biomass": "#266AA6",
        "Biogas and Charcoal": "#3B05DF",
        "Biogas and Charcoal ICS": "#3B59DF",
        "Biogas and Electricity": "#484673",
        "Electricity": '#CC503E', "Electricity and Biomass ICS": "#B497E7",
        "Electricity and LPG": "#E17C05", "Electricity and Biomass": "#FFC107",
        "Electricity and Charcoal ICS": "#660000",
        "Electricity and Biogas": "#0F8554",
        "Electricity and Charcoal": "#FF0000"}

labels = {"Biogas and Electricity": "Electricity and Biogas",
          'Collected Traditional Biomass': 'Biomass',
          'Collected Improved Biomass': 'Biomass ICS',
          'Traditional Charcoal': 'Charcoal'}

print('Creating index...')
index = {str(g): i for i, g in enumerate(africa.gdf['geometry'].unique())}
africa.gdf['index'] = [index[str(i)] for i in africa.gdf['geometry']]

print('Saving graphs...')
africa.plot_split(cmap=cmap, labels=labels, save=True, height=1.5, width=3.5)
africa.plot_costs_benefits(labels=labels, save=True, height=1.5, width=2)
africa.plot_benefit_distribution(type='box', groupby='None', cmap=cmap,
                                 labels=labels, save=True, height=1.5, width=3.5)

print('Creating map...')
africa.to_image('max_benefit_tech', name='max_benefit_tech_stats', cmap=cmap, legend_position=(0.03, 0.47),
                type='pdf', dpi=300, stats=True, stats_position=(-0.002, 0.5), stats_fontsize=10,
                labels=labels, legend=True, legend_title='Maximum benefit\ncooking technology',
                rasterized=True)

print('Saving results...')
africa.summary().to_csv(os.path.join(output_directory,
                                     f'{snakemake.params.file_name}.csv'), index=False)
africa.to_pickle('results.pkl')
