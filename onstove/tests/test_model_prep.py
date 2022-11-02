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
    assert True