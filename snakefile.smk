COUNTRIES = ['BDI']

# COUNTRIES = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR',
#              'COD', 'COG', 'DJI', 'ERI', 'ETH', 'GAB', 'GHA', 'GIN',
#              'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LSO', 'MDG', 'MLI',
#              'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'RWA', 'SDN',
#              'SEN', 'SLE', 'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TZA',
#              'UGA', 'ZAF', 'ZMB', 'ZWE']

rule all:
    input:
        expand("Africa/{country}/Demographics/Population/Population.tif", country=COUNTRIES)
        # expand(r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest/{country}/Forest.tif", country=COUNTRIES)

rule extract_forest:
    input:
         forest = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest\Forest_height_2019_SAFR.tif",
    params:
          country = "{country}"
    output:
          r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest/{country}/Forest.tif",
    script:
          "scripts/extract_forest.py"


rule process_data:
    input:
         population = "../Clean cooking Africa paper/01. Data/GIS-data/Population/{country}_ppp_2020_UNadj_constrained.tif",
         mask_layer = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp",
         # ghs =
         forest = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest/{country}/Forest.tif",
         walking_friction = r"..\Clean cooking Africa paper\01. Data\GIS-data\Walking_friction\walking_friction.tif",
         hv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\HV\All_HV.shp",
         mv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\MV\All_MV.shp",
         ntl = r"..\Clean cooking Africa paper\01. Data\GIS-data\NightLights\Africa.tif",
         traveltime_cities = r"..\Clean cooking Africa paper\01. Data\GIS-data\Traveltime_to_cities\2015_accessibility_to_cities_v2.tif",
         temperature = r"C:\Users\camilorg\Box Sync\Clean cooking Africa paper\01. Data\GIS-data\Temperature\TEMP.tif"
    params:
          output_directory = "Africa/{country}",
          country = "{country}"
    output:
          population = "Africa/{country}/Demographics/Population/Population.tif",
          forest = "Africa/{country}/Biomass/Forest/Forest.tif",
          # hv_lines = "Africa/{country}/Electricity/HV_lines/HV_lines.geojson",
          # mv_lines = "Africa/{country}/Electricity/MV_lines/MV_lines.geojson",
          # ntl = "Africa/{country}/Electricity/Night_time_lights/Night_time_lights.tif",
          # traveltime_cities = "Africa/{country}/LPG/Traveltime/Traveltime.tif",
          # temperature = "Africa/{country}/Biogas/Temperature/Temperature.tif"
    script:
          "scripts/data_processing.py"