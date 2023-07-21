import sys
from decouple import config
import os
import matplotlib.pyplot as plt

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove import OnStove

# 1. Read the OnSSTOVE model
country = snakemake.params.country
print(f'[{country}] Reading model')
model = OnStove.read_model(snakemake.input.model)

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
          'Biogas + LPG': 'LPG'
          'LPG + Biogas': 'LPG',
          'Biogas + ICS': 'ICS',
          'Electricity + LPG': "Electricity",
          "Electricity + Biogas": "Electricity",
          "Electricity + ICS": "Electricity"
          }

print(f'[{country}] Saving the graphs')
scale = int(model.base_layer.meta['width']//100*10000*2)
scale_bar_prop = dict(size=scale, style='double', textprops=dict(size=8),
                      linekw=dict(lw=1, color='black'), extent=0.01, location=(0.1, 0))
north_arrow_prop = dict(size=30, location=(0.9, 0.95), linewidth=0.5)

model.plot('max_benefit_tech', cmap=cmap,
           figsize=(16, 9),
           labels=labels, legend=True, #legend_title='Maximum benefit\ncooking technology',
           legend_position=(0.1, 0.55),
           legend_prop={'title': {'size': 10, 'weight': 'bold'}, 'size': 9},
           rasterized=True, stats=False, dpi=300,
           scale_bar=scale_bar_prop,
           north_arrow=north_arrow_prop,
           save_as='max_benefit_tech.pdf'
           )

plt.savefig(os.path.join(model.output_directory, 'max_benefit_tech.png'),
            dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(model.output_directory, 'max_benefit_tech.svg'),
            dpi=600, bbox_inches='tight', transparent=True)
plt.close()

# model.plot_split(cmap=cmap, labels=labels, save_as='tech_split.pdf', height=1.5, width=3.5)
# model.plot_costs_benefits(labels=labels, save_as='benefits_costs.pdf', height=1.5, width=2)
# model.plot_distribution(type='histogram', groupby='None', cmap=cmap, labels=labels,
#                         hh_divider=1000, y_title='Households (k)',
#                         quantiles=True,
#                         height=1.5, width=3.5, dpi=300, save_as='max_benefits_hist.pdf')
