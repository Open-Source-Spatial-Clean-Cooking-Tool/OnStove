import sys
from decouple import config
import os

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove import OnStove

# 1. Read the OnSSTOVE model
country = snakemake.params.country
print(f'[{country}] Reading model')
model = OnStove.read_model(snakemake.input.model)

# 2. Read the scenario data
print(f'[{country}] Scenario data')
path = snakemake.input.scenario_file
model.read_scenario_data(path, delimiter=',')
model.output_directory = snakemake.params.output_directory

# if snakemake.wildcards.scenario in ['Social_private_benefits']:
#     model.techs['LPG'].diesel_cost = 1.04

model.techs['Electricity'].get_capacity_cost(model)

# 3. Calculating benefits and costs of each technology and getting the max benefit technology for each cell
model.run(technologies=['Electricity', 'LPG', 'Biogas',
                        'Collected_Improved_Biomass', 'Collected_Traditional_Biomass', 'Charcoal ICS',
                        'Traditional_Charcoal', 'Biomass Forced Draft', 'Pellets Forced Draft'
                        ],
          restriction=snakemake.params.restriction)

# 5. Saving data to raster files
cmap = {"Biomass ICS (ND)": '#6F4070', "LPG": '#66C5CC', "Biomass": '#FFB6C1',
        "Biomass ICS (FD)": '#af04b3', "Pellets ICS (FD)": '#ef02f5',
        "Charcoal": '#364135', "Charcoal ICS": '#d4bdc5',
        "Biogas": '#73AF48', "Biogas and Biomass ICS (ND)": "#F6029E",
        "Biogas and Biomass ICS (FD)": "#F6029E",
        "Biogas and Pellets ICS (FD)": "#F6029E",
        "Biogas and LPG": "#0F8554", "Biogas and Biomass": "#266AA6",
        "Biogas and Charcoal": "#3B05DF",
        "Biogas and Charcoal ICS": "#3B59DF",
        "Electricity": '#CC503E', "Electricity and Biomass ICS (ND)": "#B497E7",
        "Electricity and Biomass ICS (FD)": "#B497E7",
        "Electricity and Pellets ICS (FD)": "#B497E7",
        "Electricity and LPG": "#E17C05", "Electricity and Biomass": "#FFC107",
        "Electricity and Charcoal ICS": "#660000",
        "Electricity and Biogas": "#f97b72",
        "Electricity and Charcoal": "#FF0000"}

labels = {"Biogas and Electricity": "Electricity and Biogas",
          'Collected Traditional Biomass': 'Biomass',
          'Collected Improved Biomass': 'Biomass ICS (ND)',
          'Traditional Charcoal': 'Charcoal',
          'Biomass Forced Draft': 'Biomass ICS (FD)',
          'Pellets Forced Draft': 'Pellets ICS (FD)'}

print(f'[{country}] Saving the graphs')
scale = int(model.base_layer.meta['width']//100*10000*2)
scale_bar_prop = dict(size=scale, style='double', textprops=dict(size=8),
                      linekw=dict(lw=1, color='black'), extent=0.01)
north_arow_prop = dict(size=30, location=(0.92, 0.92), linewidth=0.5)

model.to_image('max_benefit_tech', cmap=cmap, legend_position=(1, 0.75), figsize=(16, 9),
               type='pdf', dpi=300, stats=True, stats_position=(1, 0.9), stats_fontsize=10,
               labels=labels, legend=True, legend_title='Maximum benefit\ncooking technology',
               legend_prop={'title': {'size': 10, 'weight': 'bold'}, 'size': 10},
               scale_bar=scale_bar_prop, north_arrow=north_arow_prop, rasterized=True)
model.to_image('maximum_net_benefit', cmap='Spectral', cumulative_count=[0.01, 0.99],
               title=f'Maximum net benefit | {country}', dpi=300, figsize=(16, 9),
               rasterized=True, type='pdf')


model.plot_split(cmap=cmap, labels=labels, save_as='tech_split', height=1.5, width=3.5)
model.plot_costs_benefits(labels=labels, save_as='benefits_costs', height=1.5, width=2)
model.plot_benefit_distribution(type='box', groupby='None', cmap=cmap, labels=labels,
                                save_as='max_benefits_box', height=1.5, width=3.5)

print(f'[{country}] Saving the results')

model.summary().to_csv(os.path.join(snakemake.params.output_directory, 'summary.csv'), index=False)
model.to_pickle('results.pkl')
