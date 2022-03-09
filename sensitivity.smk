
# SENSITIVITY = ['VSL3_DR1_SCC1_W1']
SENSITIVITY, = glob_wildcards("../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Sensitivity_files/{sensitivity}/BDI_scenario_file.csv")

# COUNTRIES = ['BDI']

COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
             'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
             'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
             'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
             'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
             'UGA', 'ZAF', 'ZMB', 'ZWE']

rule all:
    input:
        expand("../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Results/Africa/{sensitivity}/results.pkl",
               sensitivity=SENSITIVITY)

# rule run_sens:
    # input:
         # model = "../Clean cooking Africa paper/06. Results/LPG International price - Rural-Urban/{country}/model.pkl",
         # sensitivity_file = r"..\Clean cooking Africa paper\07. Sensitivity\LPG International price - Rural-Urban\All_benefits\Sensitivity_files\{sensitivity}\{country}_scenario_file.csv"
    # params:
          # output_directory = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Results/{country}",
          # country = "{country}",
          # file_name = "{sensitivity}"
    # output:
          # summary = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Results/{country}/{sensitivity}.csv",
    # script:
          # "scripts/sensitivity.py"

rule run_sens:
    input:
         sensitivity_files = expand(r"..\Clean cooking Africa paper\07. Sensitivity\LPG International price - Rural-Urban\All_benefits\Sensitivity_files\{{sensitivity}}\{country}_scenario_file.csv",
									country=COUNTRIES),
         models = expand("../Clean cooking Africa paper/06. Results/LPG International price - Rural-Urban/{country}/model.pkl",
                          country=COUNTRIES),
         boundaries = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp"
    params:
          output_directory = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Results",
          countries = COUNTRIES,
          file_name = "{sensitivity}"
    output:
          results = "../Clean cooking Africa paper/07. Sensitivity/LPG International price - Rural-Urban/All_benefits/Results/Africa/{sensitivity}/results.pkl"
    script:
          "scripts/merge_sensitivity.py"
