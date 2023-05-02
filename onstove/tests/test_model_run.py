import os

from onstove import OnStove

def test_run_model():
    # 1. Read the OnSSTOVE model
    country = 'Rwanda'
    output_directory = os.path.join('onstove', 'tests', 'output')
    model = OnStove.read_model(os.path.join(output_directory, 'model.pkl'))

    # 2. Read the scenario data
    path = os.path.join('onstove', 'tests', 'data', 'RWA', 
                        'RWA_scenario_file.csv')
    model.read_scenario_data(path, delimiter=',')

    model.run(technologies=['Electricity', 'LPG', 'Biogas',
                            'Collected_Improved_Biomass', 
                            'Collected_Traditional_Biomass', 'Charcoal ICS',
                            'Traditional_Charcoal', 'Biomass Forced Draft', 
                            'Pellets Forced Draft'],
              restriction='Positive_Benefits')
              
    # 3. Save the results
    model.summary().to_csv(os.path.join(output_directory, 'summary.csv'), 
                           index=False)
    model.to_pickle('results.pkl')
    assert True
    