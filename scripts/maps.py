import sys
from decouple import config
import os
import matplotlib.pyplot as plt

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove import OnStove, RasterLayer

# 1. Read the OnSSTOVE model
country = snakemake.params.country
print(f'[{country}] Reading model')
model = OnStove.read_model(snakemake.input.model)
model.output_directory = snakemake.params.output_directory

# 5. Saving data to raster files
cmap = {"Electricity": '#ffc000',
        "ICS": '#00b050',
        "Biogas": '#4BACC6',
        "LPG": '#b1a0c7'}

labels = {"Biogas + Electricity": "Electricity",
          'Collected Improved Biomass': 'ICS',
          'Charcoal ICS': 'ICS',
          'Biomass Forced Draft': 'ICS',
          'Pellets Forced Draft': 'ICS',
          'Biogas + LPG': 'LPG',
          'LPG + Biogas': 'LPG',
          'Biogas + ICS': 'Biogas',
          'Electricity + LPG': "Electricity",
          "Electricity + Biogas": "Electricity",
          "Electricity + ICS": "Electricity"
          }

print(f'[{country}] Saving the graphs')
scale = int(model.base_layer.meta['width']//100*10000*2)
scale_bar_prop = dict(size=scale, style='double', textprops=dict(size=8),
                      linekw=dict(lw=1, color='black'), extent=0.01, location=(0.1, 0.05))
north_arrow_prop = dict(size=30, location=(0.9, 0.95), linewidth=0.5)

model.plot('max_benefit_tech', cmap=cmap,
           figsize=(16, 9),
           labels=labels, legend=True, legend_title='Maximum benefit\ncooking technology',
           legend_position=(1, 0.75),
           legend_prop={'title': {'size': 10, 'weight': 'bold'}, 'size': 9},
           stats=True, #stats_kwargs={'stats_position': (0.1, 0.15)},
           rasterized=True, dpi=300,
           scale_bar=scale_bar_prop,
           north_arrow=north_arrow_prop,
           save_as='max_benefit_tech.pdf'
           )

plt.savefig(os.path.join(model.output_directory, 'max_benefit_tech.png'),
            dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(model.output_directory, 'max_benefit_tech.svg'),
            dpi=600, bbox_inches='tight', transparent=True)
plt.close()

model.plot_split(cmap=cmap, labels=labels, save_as='tech_split.png', height=1.5, width=3.5)
model.plot_costs_benefits(labels=labels, save_as='benefits_costs.png', height=1.5, width=2)
model.plot_distribution(type='histogram', groupby='None', cmap=cmap, labels=labels,
                        hh_divider=1000, y_title='Households (k)',
                        quantiles=True,
                        height=1.5, width=3.5, dpi=300, save_as='max_benefits_hist.png')

model.gdf['total_annualized_costs'] = model.gdf['investment_costs'] + model.gdf['fuel_costs'] + model.gdf['om_costs'] - model.gdf['salvage_value']                        
model.to_raster('max_benefit_tech', cmap=cmap, labels=labels, nodata=0, mask=True, mask_nodata=65535)
model.to_raster('maximum_net_benefit', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('maximum_net_benefit', metric='total', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('total_annualized_costs', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('total_annualized_costs', metric='total', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('investment_costs', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('investment_costs', metric='total', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('fuel_costs', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('fuel_costs', metric='total', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('deaths_avoided', metric='per_100k', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('time_saved', metric='per_household')
model.to_raster('reduced_emissions', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('reduced_emissions', metric='total', nodata=0, mask=True, mask_nodata=-999999)
raster = RasterLayer(path=os.path.join(model.output_directory, 'Rasters/reduced_emissions_total.tif'))
raster.data[raster.data!=raster.meta['nodata']] /= 1000
raster.name = 'reduced_emissions_total'
raster.save(os.path.join(model.output_directory, 'Rasters'))
model.to_raster('health_costs_avoided', metric='per_household', nodata=0, mask=True, mask_nodata=-999999)
model.to_raster('health_costs_avoided', metric='total', nodata=0, mask=True, mask_nodata=-999999)
