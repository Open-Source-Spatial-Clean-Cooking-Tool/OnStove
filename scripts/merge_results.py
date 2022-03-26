import sys, os
import geopandas as gpd
import pandas as pd
from decouple import config

onstove_path = config('ONSSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onsstove.layer import VectorLayer
from onsstove.onsstove import OnSSTOVE

cmap = {"Biomass ICS": '#6F4070', "LPG": '#66C5CC', "Biomass": '#FFB6C1',
        "Charcoal": '#364135', "Charcoal ICS": '#d4bdc5',
        "Biogas": '#73AF48', "Biogas and Biomass ICS": "#F6029E",
        "Biogas and LPG": "#f97b72",  "Biogas and Biomass": "#266AA6",
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

df = pd.DataFrame({'country': [], 'Households': [], 'max_benefit_tech': [], 'Calibrated_pop': [],
				   'maximum_net_benefit': [], 'deaths_avoided': [], 'health_costs_avoided': [],
				   'time_saved': [], 'reduced_emissions': [], 'investment_costs': [],
				   'om_costs': [], 'fuel_costs': [], 'emissions_costs_saved': [],
				   'opportunity_cost_gained': [], 'salvage_value': [], 
				   'IsUrban': [], 'Current_elec': [], 'geometry': []})

print('Creating Africa model...')
africa = OnSSTOVE()
africa.output_directory = snakemake.params.output_directory

mask_layer = VectorLayer('admin', 'adm_1', layer_path=snakemake.input.boundaries)
mask_layer.layer = mask_layer.layer.to_crs(3857)
africa.mask_layer = mask_layer
africa.gdf = gpd.GeoDataFrame(df, crs='epsg:3857')

print('Reading country results')
for file, country in zip(snakemake.input.results, snakemake.params.countries):
    print(f'    - {country}')
    model = OnSSTOVE.read_model(file)

    model.gdf['country'] = country
    africa.gdf = africa.gdf.append(model.gdf[df.columns], ignore_index=True)

print('Creating index...')
index = {str(g): i for i, g in enumerate(africa.gdf['geometry'].unique())}
africa.gdf['index'] = [index[str(i)] for i in africa.gdf['geometry']]

# print('Saving graphs...')
# africa.plot_split(cmap=cmap, labels=labels, save=True, height=1.5, width=3.5)
# africa.plot_costs_benefits(labels=labels, save=True, height=1.5, width=2)
# africa.plot_benefit_distribution(type='box', groupby='None', cmap=cmap,
                                 # labels=labels, save=True, height=1.5, width=3.5)

# print('Creating map...')
# africa.to_image('max_benefit_tech', cmap=cmap, legend_position=(0.03, 0.47),
                # type='pdf', dpi=300, stats=True, stats_position=(-0.002, 0.5), stats_fontsize=10,
                # labels=labels, legend=True, legend_title='Maximum benefit\ncooking technology',
                # rasterized=True)

print('Saving results...')
africa.summary().to_csv('summary.csv', index=False)
africa.to_pickle('results.pkl')
