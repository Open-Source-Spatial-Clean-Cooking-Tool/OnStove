import os
cwd = os.getcwd()

SCENARIOS = ['Social_private_benefits']  # 'Private_benefits'
# SCENARIOS, = glob_wildcards("../Clean cooking Africa paper/04. OnSSTOVE inputs/LPG International price - Rural-Urban/Scenario_files/{scenario}/BDI_scenario_file.csv")
RESTRICTION = ['Positive_Benefits']

# COUNTRIES = 'Africa'

# COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#              'COD', 'COG', 'DJI', 'ERI', 'GAB', 'GHA', 'GIN',
#              'GMB', 'GNB', 'GNQ', 'LBR', 'LSO', 'MDG']

COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
             'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
             'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
             'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
             'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
             'UGA', 'ZAF', 'ZMB', 'ZWE']


rule all:
	input:
		expand("../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/results.pkl",
               country=COUNTRIES,
               scenario=SCENARIOS,
               restriction=RESTRICTION)

# rule all:
#     input:
#         expand("../Clean cooking Africa paper/06. Results/IEA/Africa/{scenario}/{restriction}/results.pkl",
#                scenario=SCENARIOS,
#                restriction=RESTRICTION)

rule extract_forest:
    input:
         forest = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest\Forest_height_2019_SAFR.tif",
    params:
          country = "{country}"
    output:
          forest = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest/{country}/Forest.tif",
    script:
          "scripts/extract_forest.py"


rule process_data:
    input:
         population = "../Clean cooking Africa paper/01. Data/GIS-data/Population/{country}_ppp_2020_UNadj_constrained.tif",
         mask_layer = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp",
         ghs = r"..\Clean cooking Africa paper\01. Data\GIS-data\Urban\Urban.tif",
         forest = rules.extract_forest.output.forest,
         walking_friction = r"..\Clean cooking Africa paper\01. Data\GIS-data\Walking_friction\walking_friction.tif",
         hv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\HV\All_HV.shp",
         mv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\MV\All_MV.gpkg",
         ntl = r"..\Clean cooking Africa paper\01. Data\GIS-data\NightLights\Africa.tif",
         traveltime_cities = r"..\Clean cooking Africa paper\01. Data\GIS-data\Traveltime_to_cities\2015_accessibility_to_cities_v2.tif",
         roads = r"..\Clean cooking Africa paper\01. Data\GIS-data\Roads/{country}/Roads.shp",
         temperature = r"..\Clean cooking Africa paper\01. Data\GIS-data\Temperature\Temperature.tif",
         buffaloes = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\buffaloes\buffaloes.tif",
         cattles = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\cattles\cattles.tif",
         poultry = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\poultry\poultry.tif",
         goats = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\goats\goats.tif",
         pigs = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\pigs\pigs.tif",
         sheeps = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\sheeps\sheeps.tif",
         water = r"..\Clean cooking Africa paper\01. Data\GIS-data\Water scarcity\y2019m07d11_aqueduct30_annual_v01.gpkg"
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/Processed data/{country}",
          country = "{country}"
    output:
          mask_layer = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Administrative/Country_boundaries/Country_boundaries.geojson",
          population = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Demographics/Population/Population.tif",
          ghs = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Demographics/Urban/Urban.tif",
          forest = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biomass/Forest/Forest.tif",
          biomass_friction = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biomass/Friction/Friction.tif",
          # hv_lines = "Africa/{country}/Electricity/HV_lines/HV_lines.geojson",
          mv_lines = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Electricity/MV_lines/MV_lines.geojson",
          ntl = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Electricity/Night_time_lights/Night_time_lights.tif",
          traveltime_cities = "../Clean cooking Africa paper/06. Results/Processed data/{country}/LPG/Traveltime/Traveltime.tif",
          roads = "../Clean cooking Africa paper/06. Results/Processed data/{country}/LPG/Roads/roads.geojson",
          temperature = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Temperature/Temperature.tif",
          buffaloes = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/buffaloes/buffaloes.tif",
          cattles = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/cattles/cattles.tif",
          poultry = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/poultry/poultry.tif",
          goats = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/goats/goats.tif",
          pigs = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/pigs/pigs.tif",
          sheeps = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Livestock/sheeps/sheeps.tif",
          water = "../Clean cooking Africa paper/06. Results/Processed data/{country}/Biogas/Water scarcity/Water scarcity.tif"
    script:
          "scripts/data_processing.py"

rule prepare_model:
   input:
        prep_file = "../Clean cooking Africa paper/04. OnSSTOVE inputs/IEA/Prep_files/{country}_prep_file.csv",
        techs_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\IEA\Technical_specs\{scenario}\{country}_file_tech_specs.csv",
        mask_layer = rules.process_data.output.mask_layer,
        population = rules.process_data.output.population,
        ghs = rules.process_data.output.ghs,
        forest = rules.process_data.output.forest,
        biomass_friction = rules.process_data.output.biomass_friction,
        mv_lines = rules.process_data.output.mv_lines,
        ntl = rules.process_data.output.ntl,
        traveltime_cities = rules.process_data.output.traveltime_cities,
        roads = rules.process_data.output.roads,
        temperature = rules.process_data.output.temperature,
        water = rules.process_data.output.water,
        buffaloes = rules.process_data.output.buffaloes,
        cattles = rules.process_data.output.buffaloes,
        poultry = rules.process_data.output.poultry,
        goats = rules.process_data.output.goats,
        pigs = rules.process_data.output.pigs,
        sheeps = rules.process_data.output.sheeps
   params:
         wealth_index = r"..\Clean cooking Africa paper\01. Data\GIS-data\Poverty\{country}_relative_wealth_index",
         output_directory = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}",
         country = "{country}"
   output:
         model = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/model.pkl"
   script:
         "scripts/model_preparation.py"

rule run_model:
   input:
        model = rules.prepare_model.output.model,
        scenario_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\IEA\Scenario_files\{scenario}\{country}_scenario_file.csv"
   params:
         output_directory = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}",
         country = "{country}",
         restriction = "{restriction}"
   output:
         results = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/results.pkl",
   script:
        "scripts/model_run.py"

rule main_plot:
    input:
         tech_split = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/tech_split.pdf",
         max_benefits = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/max_benefits_hist.pdf",
         benefits_costs = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/benefits_costs.pdf",
         max_benefit_tech = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/max_benefit_tech.pdf"
    params:
          path = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/",
    output:
          main_plot = "../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/main_plot.pdf"
    shell:
         """
         pdflatex "\\newcommand\path{{"{params.path}"}}\input{{scripts/main_plot.tex}}" -output-directory "{params.path}"
         """

rule get_main_plot:
   input:
        expand("../Clean cooking Africa paper/06. Results/IEA/{country}/{scenario}/{restriction}/main_plot.pdf",
               country=COUNTRIES,
               scenario=SCENARIOS,
               restriction=RESTRICTION)


rule merge_results:
   input:
        results = expand("../Clean cooking Africa paper/06. Results/IEA/{country}/{{scenario}}/{{restriction}}/results.pkl",
                         country=COUNTRIES),
        boundaries = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp"
   params:
         output_directory = "../Clean cooking Africa paper/06. Results/IEA/Africa/{scenario}/{restriction}",
         countries = COUNTRIES
   output:
         results = "../Clean cooking Africa paper/06. Results/IEA/Africa/{scenario}/{restriction}/results.pkl"
   script:
         "scripts/merge_results.py"