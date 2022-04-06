
# SENSITIVITY = ['1']
SENSITIVITY, = glob_wildcards("../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/Sensitivity_files/{sensitivity}/Scenario_files/BDI_scenario_file.csv")

# COUNTRIES = ['BDI']

COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
             'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
             'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
             'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
             'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
             'UGA', 'ZAF', 'ZMB', 'ZWE']

rule all:
    input:
        expand("../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/Results/Africa/{sensitivity}.csv",
               sensitivity=SENSITIVITY)

rule run_sens:
    input:
         sensitivity_files = expand(r"..\Clean cooking Africa paper\07. Sensitivity\LPG International price - Rural-Urban/Sensitivity_files/{{sensitivity}}/Scenario_files/{country}_scenario_file.csv",
                                    country=COUNTRIES),
         technology_file = r"..\Clean cooking Africa paper\07. Sensitivity\LPG International price - Rural-Urban\Sensitivity_files/{sensitivity}/Technical_specs\Tech_specs.csv",
         models = expand("../Clean cooking Africa paper/06. Results/LPG International price - Rural-Urban/{country}/model.pkl",
                          country=COUNTRIES),
         # boundaries = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp"
    params:
          output_directory = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/Results",
          countries = COUNTRIES,
          file_name = "{sensitivity}"
    output:
          results = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/Results/Africa/{sensitivity}.csv"
    script:
          "scripts/merge_sensitivity.py"
