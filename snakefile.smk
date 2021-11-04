COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
             'COD', 'COG', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
             'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
             'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA',
             'SEN', 'SLE', 'SWZ', 'TCD', 'TGO', 'TZA',
             'UGA', 'ZAF', 'ZMB', 'ZWE']

# COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#              'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
#              'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
#              'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
#              'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
#              'UGA', 'ZAF', 'ZMB', 'ZWE']

rule all:
    input:
        expand("../Clean cooking Africa paper/06. Results/{country}/Output/results.csv", country=COUNTRIES)
        # expand(r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest/{country}/Forest.tif", country=COUNTRIES)

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
         temperature = r"..\Clean cooking Africa paper\01. Data\GIS-data\Temperature\TEMP.tif"
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/{country}",
          country = "{country}"
    output:
          mask_layer = "../Clean cooking Africa paper/06. Results/{country}/Administrative/Country_boundaries/Country_boundaries.geojson",
          population = "../Clean cooking Africa paper/06. Results/{country}/Demographics/Population/Population.tif",
          ghs = "../Clean cooking Africa paper/06. Results/{country}/Demographics/Urban/Urban.tif",
          forest = "../Clean cooking Africa paper/06. Results/{country}/Biomass/Forest/Forest.tif",
          biomass_friction = "../Clean cooking Africa paper/06. Results/{country}/Biomass/Friction/Friction.tif",
          # hv_lines = "Africa/{country}/Electricity/HV_lines/HV_lines.geojson",
          mv_lines = "../Clean cooking Africa paper/06. Results/{country}/Electricity/MV_lines/MV_lines.geojson",
          ntl = "../Clean cooking Africa paper/06. Results/{country}/Electricity/Night_time_lights/Night_time_lights.tif",
          traveltime_cities = "../Clean cooking Africa paper/06. Results/{country}/LPG/Traveltime/Traveltime.tif",
          temperature = "../Clean cooking Africa paper/06. Results/{country}/Biogas/Temperature/Temperature.tif"
    script:
          "scripts/data_processing.py"

rule prepare_model:
    input:
         prep_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\Prep_files\{country}_prep_file.csv",
         wealth_index = r"..\Clean cooking Africa paper\01. Data\GIS-data\Poverty\{country}_relative_wealth_index.csv" ,
         techs_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\Technical specs\{country}_file_tech_specs.csv",
         mask_layer = rules.process_data.output.mask_layer,
         population = rules.process_data.output.population,
         ghs = rules.process_data.output.ghs,
         forest = rules.process_data.output.forest,
         biomass_friction = rules.process_data.output.biomass_friction,
         # hv_lines = rules.process_data.output.hv_lines,
         mv_lines = rules.process_data.output.mv_lines,
         ntl = rules.process_data.output.ntl,
         traveltime_cities = rules.process_data.output.traveltime_cities,
         temperature = rules.process_data.output.temperature,
         water = r"..\Clean cooking Africa paper\01. Data\GIS-data\Water scarcity\y2019m07d11_aqueduct30_annual_v01.gpkg",
         tiers = r"..\Clean cooking Africa paper\01. Data\GIS-data\Electricity tiers\tiersofaccess_SSA_2018.tif"
    params:
          output_directory = "../Clean cooking Africa paper/06. Results/{country}",
          country = "{country}"
    output:
          model = "../Clean cooking Africa paper/06. Results/{country}/model.pkl"
    script:
          "scripts/model_preparation.py"

rule run_model:
    input:
         model = rules.prepare_model.output.model,
         scenario_file = r"..\Clean cooking Africa paper\04. OnSSTOVE inputs\Scenario_files\{country}_scenario_file.csv"
    params:
          country = "{country}"
    output:
          results = "../Clean cooking Africa paper/06. Results/{country}/Output/results.csv",
    script:
          "scripts/model_run.py"