import sys
from decouple import config
import os
import pandas as pd

onstove_path = config('ONSTOVE').format(os.getlogin())
sys.path.append(onstove_path)

from onstove.onstove import OnStove

def run_model(country, model_file, sensitivity_file, tech_file, output_directory, file_name):
	# 1. Read the OnSSTOVE model
	print(f'[{country}] Reading model')
	print(model_file)
	model = OnStove.read_model(model_file)

	# 2. Read the scenario data
	print(f'[{country}] Scenario data')
	path = sensitivity_file
	print(path)
	model.read_scenario_data(path, delimiter=',')
	tech_specs = pd.read_csv(tech_file).T.to_dict()
	for value in tech_specs.values():
		model.techs[value['Fuel']][value['Param']] = value['Value']
	model.output_directory = output_directory
	model.techs['Electricity'].get_capacity_cost(model)

	# 3. Calculating benefits and costs of each technology and getting the max benefit technology for each cell
	model.run(technologies=['Electricity', 'LPG', 'Biogas',
							'Collected_Improved_Biomass', 'Collected_Traditional_Biomass', 'Charcoal ICS',
							'Traditional_Charcoal'
							])

	# 4. Save the summary
	os.makedirs(output_directory, exist_ok=True)
	model.summary().to_csv(os.path.join(output_directory,
									    f'{file_name}.csv'),
						   index=False)
										
	return model
