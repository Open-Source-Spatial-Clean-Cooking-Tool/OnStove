from matplotlib.colors import to_rgb
import pandas as pd

print('Reading results')
df = pd.read_csv(snakemake.input.results)

print('Creating colormap')
dff = df.groupby('max_benefit_tech_code').agg({'max_benefit_tech': 'first'})

cmap = {"ICS": '#57365A', "LPG": '#6987B7', "Traditional biomass": '#673139', "Charcoal": '#B6195E',
        "Biogas": '#3BE2C5', "Biogas and ICS": "#F6029E",
        "Biogas and LPG": "#C021C0", "Biogas and Traditional biomass": "#266AA6",
        "Biogas and Charcoal": "#3B05DF", "Biogas and Electricity": "#484673",
        "Electricity": '#D0DF53', "Electricity and ICS": "#4D7126",
        "Electricity and LPG": "#004D40", "Electricity and Traditional biomass": "#FFC107",
        "Electricity and Charcoal": "#1E88E5", "Electricity and Biogas": "#484673"}

with open(snakemake.output.colormap, 'w') as f:
    for index, row in dff.iterrows():
        r = int(to_rgb(cmap[row['max_benefit_tech']])[0] * 255)
        g = int(to_rgb(cmap[row['max_benefit_tech']])[1] * 255)
        b = int(to_rgb(cmap[row['max_benefit_tech']])[2] * 255)
        f.write(f'{index} {r} {g} {b} 255 {row["max_benefit_tech"]}\n')
