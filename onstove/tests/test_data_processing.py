import os
import numpy as np

from onstove import DataProcessor, VectorLayer, RasterLayer

def test_process_data():
    # 1. Create a data processor
    output_directory = os.path.join('onstove', 'tests', 'output')
    country = 'Rwanda'
    data = DataProcessor(project_crs=3857, cell_size=(1000, 1000))
    data.output_directory = output_directory
    assert isinstance(data, DataProcessor)
    
    # 2. Add a mask layer (country boundaries)
    adm_path = os.path.join('onstove', 'tests', 'data', 'RWA', 'Administrative', 
                            'Country_boundaries', 'Country_boundaries.geojson')
    data.add_mask_layer(category='Administrative', name='Country_boundaries',
                        path=adm_path)
    assert isinstance(data.mask_layer, VectorLayer)

    # 3. Add GIS layers

    # Demographics
    ## Population
    pop_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                            'Demographics', 'Population', 'Population.tif')
    data.add_layer(category='Demographics', name='Population', path=pop_path,
                   layer_type='raster', base_layer=True, resample='sum')
    assert isinstance(data.layers['Demographics']['Population'], RasterLayer)
    
    ## GHS
    ghs_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                            'Demographics', 'Urban', 'Urban.tif')
    data.add_layer(category='Demographics', name='Urban', path=ghs_path, 
                   layer_type='raster', resample='nearest')
    assert isinstance(data.layers['Demographics']['Urban'], RasterLayer)
    
    # Biomass
    ## forest
    forest_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                               'Biomass', 'Forest', 'Forest.tif')
    data.add_layer(category='Biomass', name='Forest', path=forest_path, 
                   layer_type='raster', resample='sum')
    data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data < 5] = 0
    data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data >= 5] = 1
    data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
    transform = data.layers['Biomass']['Forest'].calculate_default_transform(data.project_crs)[0]
    factor = (data.cell_size[0] ** 2) / (transform[0] ** 2)
    assert isinstance(data.layers['Biomass']['Forest'], RasterLayer)
    
    ## Friction
    friction_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                                 'Biomass', 'Friction', 'Friction.tif')
    data.add_layer(category='Biomass', name='Friction', path=friction_path, 
                   layer_type='raster', resample='average', window=False)
    assert isinstance(data.layers['Biomass']['Friction'], RasterLayer)
    
    # Electricity
    ## MV lines
    mv_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                           'Electricity', 'MV_lines', 'MV_lines.geojson')
    data.add_layer(category='Electricity', name='MV_lines', path=mv_path,
                   layer_type='vector', window=False)
    assert isinstance(data.layers['Electricity']['MV_lines'], VectorLayer)

    ## NTL
    ntl_path = os.path.join('onstove', 'tests', 'data', 'RWA', 'Electricity', 
                            'Night_time_lights', 'Night_time_lights.tif')
    data.add_layer(category='Electricity', name='Night_time_lights', 
                   path=ntl_path, layer_type='raster',
                   resample='average', window=False)
    data.layers['Electricity']['Night_time_lights'].save(f'{data.output_directory}/Electricity/Night_time_lights')
    assert isinstance(data.layers['Electricity']['Night_time_lights'], 
                      RasterLayer)
    
    # LPG
    ## Traveltime
    traveltime_cities = os.path.join('onstove', 'tests', 'data', 'RWA', 
                                     'LPG', 'Traveltime', 'Traveltime.tif')
    data.add_layer(category='LPG', name='Traveltime', path=traveltime_cities,
                   layer_type='raster', resample='average', window=False)
    data.layers['LPG']['Traveltime'].save(f'{data.output_directory}/LPG/Traveltime')
    assert isinstance(data.layers['LPG']['Traveltime'], RasterLayer)
    
    # Biogas
    ## Temperature
    temperature = os.path.join('onstove', 'tests', 'data', 'RWA', 
                               'Biogas', 'Temperature', 'Temperature.tif')
    data.add_layer(category='Biogas', name='Temperature', path=temperature,
                   layer_type='raster', resample='average', window=False)
    data.layers['Biogas']['Temperature'].save(f'{data.output_directory}/Biogas/Temperature')
    data.mask_layers(datasets={'Biogas': ['Temperature']})
    assert isinstance(data.layers['Biogas']['Temperature'], RasterLayer)
    
    ## Livestock
    buffaloes = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                             'Livestock', 'buffaloes', 'buffaloes.tif')
    cattles = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                           'Livestock', 'cattles', 'cattles.tif')
    poultry = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                           'Livestock', 'poultry', 'poultry.tif')
    goats = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                         'Livestock', 'goats', 'goats.tif')
    pigs = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                        'Livestock', 'pigs', 'pigs.tif')
    sheeps = os.path.join('onstove', 'tests', 'data', 'RWA', 'Biogas', 
                          'Livestock', 'sheeps', 'sheeps.tif')

    for key, path in {'buffaloes': buffaloes,
                 'cattles': cattles,
                 'poultry': poultry,
                 'goats': goats,
                 'pigs': pigs,
                 'sheeps': sheeps}.items():
        data.add_layer(category='Biogas/Livestock', name=key, path=path,
                       layer_type='raster', resample='nearest', window=False, 
                       rescale=True)
        assert isinstance(data.layers['Biogas/Livestock'][key], RasterLayer)
        
    ## Water scarcity
    water_path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                              'Biogas', 'Water scarcity', 'Water scarcity.gpkg')
    water = VectorLayer(category='Biogas', name='Water scarcity', 
                        path=water_path, bbox=data.mask_layer.data)
    water.data["class"] = 0
    water.data['class'] = np.where(water.data['bws_label'].isin([
                                                         'Low (<10%)',
                                                         'Low - Medium (10-20%)'
                                                         ]), 1, 0)
    water.data.to_crs(data.project_crs, inplace=True)
    out_folder = os.path.join(data.output_directory, "Biogas", "Water scarcity")
    water.rasterize(cell_height=data.cell_size[0], cell_width=data.cell_size[1],
                    attribute="class", output_path=out_folder, nodata=0)
    data.add_layer(category='Biogas', name='Water scarcity',
                   path=os.path.join(out_folder, 'Water scarcity.tif'),
                   layer_type='raster', resample='nearest')
    assert isinstance(data.layers['Biogas']['Water scarcity'], RasterLayer)
    
    # 4. Mask reproject and align all required layers
    ## Align all rasters
    data.align_layers(datasets='all')

    ## mask all rasters
    data.mask_layers(datasets='all')

    ## Reprojects all rasters
    data.reproject_layers(datasets={'Electricity': ['MV_lines']})

    ## Calculate canopy cover
    data.layers['Biomass']['Forest'].data /= factor
    data.layers['Biomass']['Forest'].data *= 100
    data.layers['Biomass']['Forest'].data[data.layers['Biomass']['Forest'].data > 100] = 100
    data.layers['Biomass']['Forest'].save(f'{data.output_directory}/Biomass/Forest')
    assert True