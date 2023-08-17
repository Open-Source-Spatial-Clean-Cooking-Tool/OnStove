import os
cwd = os.getcwd()

SCENARIOS = ['Social_private_benefits']
RESTRICTION = ['Positive_Benefits']

COUNTRIES = ['KEN']
# COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#              'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
#              'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
#              'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
#              'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
#              'UGA', 'ZAF', 'ZMB', 'ZWE']


rule all:
    input:
        expand("../Clean cooking Africa paper/06. Results/IEA/Archive/{country}/{scenario}/{restriction}/max_benefit_tech.pdf",
               country=COUNTRIES,
               scenario=SCENARIOS,
               restriction=RESTRICTION)
               
rule get_maps:
    input:
        model = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/results.pkl"
    params:
        country = "{country}",
        output_directory = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}"
    output:
        "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/max_benefit_tech.pdf"
    script:
        "scripts/maps.py"

