SCENARIOS = ['LPG International price - Rural-Urban']
SENSITIVITY, = glob_wildcards("../Clean cooking Africa paper/04. OnSSTOVE inputs/LPG International price - Rural-Urban/Sensitivity_files/{sensitivity}/BDI_scenario_file.csv")

COUNTRIES = ['RWA']

#COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#             'COD', 'COG', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
#             'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
#             'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA',
#             'SEN', 'SLE', 'SWZ', 'TCD', 'TGO', 'TZA',
#             'UGA', 'ZAF', 'ZMB', 'ZWE']

# COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#              'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
#              'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
#              'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
#              'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
#              'UGA', 'ZAF', 'ZMB', 'ZWE']

rule all:
    input:
        expand("../Clean cooking Africa paper/06. Results/{scenario}/{country}/{sensitivity}/Output/results.pkl",
               country=COUNTRIES,
               sensitivity=SENSITIVITY,
               scenario=SCENARIOS)

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
         ghs = r"..\Clean cooking Africa paper\01. Data\GIS-data\Urban\GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0.tif",
         forest = rules.extract_forest.output.forest,
         walking_friction = r"..\Clean cooking Africa paper\01. Data\GIS-data\Walking_friction\walking_friction.tif",
         hv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\HV\All_HV.shp",
         mv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\MV\All_MV.gpkg",
         ntl = r"..\Clean cooking Africa paper\01. Data\GIS-data\NightLights\Africa.tif",
         traveltime_cities = r"..\Clean cooking Africa paper\01. Data\GIS-data\Traveltime_to_cities\2015_accessibility_to_cities_v2.tif",
         temperature = r"..\Clean cooking Africa paper\01. Data\GIS-data\Temperature\TEMP.tif",
         buffaloes = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Buffaloes\5_Bf_2010_Da.tif",
         cattles = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Cattle\5_Ct_2010_Da.tif",
         poultry = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Chickens\5_Ch_2010_Da.tif",
         goats = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Goats\5_Gt_2010_Da.tif",
         pigs = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Pigs\5_Pg_2010_Da.tif",
         sheeps = r"..\Clean cooking Africa paper\01. Data\GIS-data\Global livestock\Sheep\5_Sh_2010_Da.tif",
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
         prep_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\{scenario}\Prep_files\{country}_prep_file.csv",
         wealth_index = r"..\Clean cooking Africa paper\01. Data\GIS-data\Poverty\{country}_relative_wealth_index.csv" ,
         techs_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\{scenario}\Technical_specs\{country}_file_tech_specs.csv",
         mask_layer = rules.process_data.output.mask_layer,
         population = rules.process_data.output.population,
         ghs = rules.process_data.output.ghs,
         forest = rules.process_data.output.forest,
         biomass_friction = rules.process_data.output.biomass_friction,
         mv_lines = rules.process_data.output.mv_lines,
         ntl = rules.process_data.output.ntl,
         traveltime_cities = rules.process_data.output.traveltime_cities,
         temperature = rules.process_data.output.temperature,
         water = rules.process_data.output.water,
         tiers = r"..\Clean cooking Africa paper\01. Data\GIS-data\Electricity tiers\tiersofaccess_SSA_2018.tif",
         buffaloes = rules.process_data.output.buffaloes,
         cattles = rules.process_data.output.buffaloes,
         poultry = rules.process_data.output.poultry,
         goats = rules.process_data.output.goats,
         pigs = rules.process_data.output.pigs,
         sheeps = rules.process_data.output.sheeps
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/{scenario}/{country}",
          country = "{country}"
    output:
          model = "../Clean cooking Africa paper/06. Results/{scenario}/{country}/model.pkl"
    script:
          "scripts/model_preparation.py"

rule run_model:
    input:
         model = rules.prepare_model.output.model,
         scenario_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\{scenario}\Sensitivity_files\{sensitivity}\{country}_scenario_file.csv"
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/{scenario}/{country}/{sensitivity}",
          country = "{country}"
    output:
          results = "../Clean cooking Africa paper/06. Results/{scenario}/{country}/{sensitivity}/Output/results.pkl",
    script:
          "scripts/model_run.py"

rule merge_results:
    input:
         results = expand("../Clean cooking Africa paper/06. Results/{{scenario}}/{country}/{{sensitivity}}/Output/results.csv",
                           country=COUNTRIES),
         summaries = expand("../Clean cooking Africa paper/06. Results/{{scenario}}/{country}/{{sensitivity}}/Output/summary.csv",
                            country=COUNTRIES),
         boundaries = rules.process_data.input.mask_layer
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/Africa/{scenario}/{sensitivity}",
          countries = COUNTRIES
    output:
          map = "../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/max_benefit_tech.tif",
          summary = "../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/summary.csv",
          results = "../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/results.gz"
    script:
          "scripts/merge_results.py"

rule get_map:
    input:
         expand("../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/max_benefit_tech.tif",
               sensitivity=SENSITIVITY,
               scenario=SCENARIOS)

rule colormap:
    input:
         results = rules.merge_results.output.results
    output:
          colormap = "../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/ColorMap.clr"
    script:
          "scripts/colormap_generator.py"

rule get_colormap:
    input:
         expand("../Clean cooking Africa paper/06. Results/{scenario}/Africa/{sensitivity}/ColorMap.clr",
               sensitivity=SENSITIVITY,
               scenario=SCENARIOS)