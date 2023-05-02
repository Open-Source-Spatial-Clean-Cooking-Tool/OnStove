import os

from onstove import VectorLayer, RasterLayer, OnStove

def test_prepare_model():
    # 1. Create an OnStove model
    output_directory = os.path.join('onstove', 'tests', 'output')
    country = 'Rwanda'
    model = OnStove()
    model.output_directory = output_directory
    assert isinstance(model, OnStove)
    
    # 2. Read the model data
    path = os.path.join('onstove', 'tests', 'data', 'RWA', 'RWA_prep_file.csv')
    model.read_scenario_data(path, delimiter=',')
    
    # 3. Add a country mask layer
    path = os.path.join(output_directory, 'Administrative', 
                        'Country_boundaries', 'Country_boundaries.geojson')
    mask_layer = VectorLayer('admin', 'adm_1', path=path)
    model.mask_layer = mask_layer

    # 4. Add a population base layer
    path = os.path.join(output_directory, 'Demographics', 
                        'Population', 'Population.tif')
    model.add_layer(category='Demographics', name='Population', path=path, 
                    layer_type='raster', base_layer=True)
    model.population_to_dataframe()
    
    # 5. Calibrate population and urban/rural split
    ghs_path = os.path.join(output_directory, 'Demographics', 
                            'Urban', 'Urban.tif')
    model.calibrate_urban_rural_split(ghs_path)
    
    # 6. Add wealth index GIS data
    wealth_index = os.path.join('onstove', 'tests', 'data', 'RWA', 
                                'Demographics', 'Wealth', 
                                'RWA_relative_wealth_index')
    if country in ['SOM', 'SDN', 'SSD']:
        model.extract_wealth_index(wealth_index + '.shp', file_type='polygon',
                                   x_column="longitude", y_column="latitude", 
                                   wealth_column="rwi")
    else:
        model.extract_wealth_index(wealth_index + '.csv', file_type='csv',
                                   x_column="longitude", y_column="latitude", 
                                   wealth_column="rwi")
    
    # 7. Read electricity network GIS layers
    ## Read MV lines
    path = os.path.join(output_directory, 'Electricity', 
                        'MV_lines', 'MV_lines.geojson')
    mv_lines = VectorLayer('Electricity', 'MV_lines', path=path, 
                           distance_method='proximity')

    ## Calculate distance to electricity infrastructure
    model.distance_to_electricity(mv_lines=mv_lines)

    ## Add night time lights data
    path = os.path.join(output_directory, 'Electricity', 
                        'Night_time_lights', 'Night_time_lights.tif')
    ntl = RasterLayer('Electricity', 'Night_time_lights', path=path)

    model.raster_to_dataframe(ntl, name='Night_lights', method='read',
                              fill_nodata_method='interpolate')
                              
    # 8. Calibrate current electrified population
    model.final_elec()
    rate = model.gdf['Elec_pop_calib'].sum() / model.gdf['Calibrated_pop'].sum()
    assert abs(rate - model.specs['elec_rate']) <= 0.05
    
    # 9. Read the cooking technologies data
    path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                        'RWA_file_tech_specs.csv')
    model.read_tech_data(path, delimiter=',')
    
    # 10. Reading GIS data for LPG supply
    path = os.path.join(output_directory, 'LPG', 
                        'Traveltime', 'Traveltime.tif')
    travel_time = RasterLayer('LPG', 'Traveltime', path=path)
    model.techs['LPG'].travel_time = model.raster_to_dataframe(travel_time,
                                            fill_nodata_method='interpolate',
                                            method='read') * 2 / 60
                                            
    # 11. Adding GIS data for Traditional Biomass
    friction_path = os.path.join(output_directory, 'Biomass', 
                                 'Friction', 'Friction.tif')
    forest_path = os.path.join(output_directory, 'Biomass', 
                               'Forest', 'Forest.tif')
    model.techs['Collected_Traditional_Biomass'].friction_path = friction_path
    model.techs['Collected_Traditional_Biomass'].forest_path = forest_path
    model.techs['Collected_Traditional_Biomass'].forest_condition = lambda x: x > 0

    # 12. Adding GIS data for Improved Biomass (ICS biomass)
    model.techs['Collected_Improved_Biomass'].friction_path = friction_path
    model.techs['Collected_Improved_Biomass'].forest_path = forest_path
    model.techs['Collected_Improved_Biomass'].forest_condition = lambda x: x > 0

    model.techs['Biomass Forced Draft'].friction_path = friction_path
    model.techs['Biomass Forced Draft'].forest_path = forest_path
    model.techs['Biomass Forced Draft'].forest_condition = lambda x: x > 0
    
    # 13. Adding GIS data for Improved Biomass collected (ICS biomass)
    if 'Biogas' in model.techs.keys():
        ## Adding livestock data
        buffaloes = os.path.join(output_directory, 'Biogas', 'Livestock',
                                 'buffaloes', 'buffaloes.tif')
        cattles = os.path.join(output_directory, 'Biogas', 'Livestock',
                               'cattles', 'cattles.tif')
        poultry = os.path.join(output_directory, 'Biogas', 'Livestock',
                               'poultry', 'poultry.tif')
        goats = os.path.join(output_directory, 'Biogas', 'Livestock',
                             'goats', 'goats.tif')
        pigs = os.path.join(output_directory, 'Biogas', 'Livestock',
                            'pigs', 'pigs.tif')
        sheeps = os.path.join(output_directory, 'Biogas', 'Livestock',
                              'sheeps', 'sheeps.tif')
        path_temp = os.path.join(output_directory, 'Biogas', 'Temperature', 
                                 'Temperature.tif')
        path_water = os.path.join(output_directory, 'Biogas', 'Water scarcity', 
                                  'Water scarcity.tif')
        model.techs['Biogas'].temperature = RasterLayer('Biogas', 'Temperature', 
                                                        path_temp)
        model.techs['Biogas'].water = RasterLayer('Biogas', 'Water scarcity', 
                                                  path_water)

        ## Recalibrating livestock
        model.techs['Biogas'].recalibrate_livestock(model, buffaloes,
                                                    cattles, poultry, goats, 
                                                    pigs, sheeps)
        model.techs['Biogas'].friction_path = friction_path

    # 14. Saving the prepared model inputs
    model.to_pickle("model.pkl")
    assert True
