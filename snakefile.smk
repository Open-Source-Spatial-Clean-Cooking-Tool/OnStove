
COUNTRIES = ['AGO']

rule all:
    input:
        expand("results/{country}/Demographics/Population/Population.tif", country=COUNTRIES)

rule process_data:
    input:
         population = "../Clean cooking Africa paper/01. Data/GIS-data/Population/{country}_ppp_2020_UNadj_constrained.tif",
         mask_layer = r"..\Clean cooking Africa paper\01. Data\GIS-data\Admin\Admin_1.shp",
         # ghs =
         forest = r"..\Clean cooking Africa paper\01. Data\GIS-data\Forest\Forest_height_2019_SAFR.tif",
         walking_friction = r"..\Clean cooking Africa paper\01. Data\GIS-data\Walking_friction\walking_friction.tif",
         hv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\HV\All_HV.shp",
         mv_lines = r"..\Clean cooking Africa paper\01. Data\GIS-data\MV\All_MV.shp",
         ntl = r"..\Clean cooking Africa paper\01. Data\GIS-data\NightLights\Africa.tif",
         travetime_cities = r"..\Clean cooking Africa paper\01. Data\GIS-data\Traveltime_to_cities\2015_accessibility_to_cities_v2.tif",
         temperature = r"C:\Users\camilorg\Box Sync\Clean cooking Africa paper\01. Data\GIS-data\Temperature\TEMP.tif"
    params:
          output_directory = "{country}"
    output:
          population = "results/{country}/Demographics/Population/Population.tif",
          forest = "results/{country}/Biomass/Forest/Forest.tif",
          hv_lines = "results/{country}/Electricity/HV_lines/HV_lines.geojson",
          mv_lines = "results/{country}/Electricity/MV_lines/MV_lines.geojson",
          ntl = "results/{country}/Electricity/Night_time_lights/Night_time_lights.tif",
          traveltime_cities = "results/{country}/LPG/Traveltime/Traveltime.tif",
          temperature = "results/{country}/Biogas/Temperature/Temperature.tif"
    script:
          "scripts/data_processing.py"