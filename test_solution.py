import json
from argparse import ArgumentParser

DATA_TYPE_LIST = 'ntic'

def getSolutionScore(solution_txt,dataset_txt, show_details = True):
	"""Evaluate the solution and return a tuple (score, is_valid, message).

	Arguments:
	solution_txt -- the solution to be evaluated (as a string)
	dataset_txt -- the dataset for which the solution is made (as a string)

	show_details -- an optional parameter to show some details on the submitted solution
	"""

	try:
		dataset = json.loads(dataset_txt)
	except:
		return 0, False, 'Error while processing the dataset. Please contact the contest organizer.'
	
	try:
		solution = json.loads(solution_txt)
	except:
		return 0, False, 'Submission is not a valid JSON object'

	for key in ('dataMappings','modelMappings'):
		if key not in solution:
			return 0, False, f'Submission does not contain the "{key}" key'
		if not isinstance(solution[key], list):
			return 0, False, f'"{key}" is not a list'
		for mapping in solution[key]:
			if not isinstance(mapping,list) or len(mapping)!=2 or not isinstance(mapping[0],int) or not isinstance(mapping[1],int):
				return 0, False, f'One item of "{key}" does not have the right format'

	model_used = set()
	model_used.update(model for _,model in solution["dataMappings"])
	model_used.update(model for models in solution["modelMappings"] for model in models)

	final_value = final_energy = 0

	model_fills = {i:[0]*4 for i in model_used}
	model_defiled ={i:0 for i in model_used}

	data_passed = set()

	for db,model in solution["dataMappings"]:
		if not 0<=db<len(dataset["datasets"]):
			return 0, False, f"Data {db} does not exist"
		if not 0<=model<len(dataset["models"]):
			return 0, False, f"Model {model} does not exist"

		if db in data_passed:
			return 0, False, f"Data {db} has been assigned more than once"
		data_passed.add(db)

		data_type = dataset["datasets"][db]["dataType"]

		if data_type not in dataset["models"][model]["dataRequirements"]:
			return 0, False, f"Model {model} does not accept the type of dataset {db} ({data_type})"

		data_type_to_number = DATA_TYPE_LIST.find(dataset["datasets"][db]["dataType"])
		model_fills[model][data_type_to_number]+=dataset["datasets"][db]["size"]
		model_defiled[model]|=dataset["datasets"][db]["isCopyrighted"]

	model_used_as_source = set()

	for model_source, model_target in solution["modelMappings"]:
		if model_source in model_used_as_source:
			return 0, False, f"Model {model_source} has already been used as a data source"
		model_used_as_source.add(model_source)
		if len(dataset["models"][model_source]["dataRequirements"])!=1:
			return 0, False, f"Model {model_source}, used as a data source, needs more than one data type"
		data_type = next(iter(dataset["models"][model_source]["dataRequirements"]))
		data_type_to_number = DATA_TYPE_LIST.find(data_type)

		data_req = dataset["models"][model_source]["dataRequirements"][data_type]
		if not data_req["lowerBound"]<=model_fills[model_source][data_type_to_number]<=data_req["upperBound"]:
			return 0, False, f"The data ({data_type}) requirements of model {model_source} used as data source are not fulfilled"

		model_defiled[model_target]=1
		model_fills[model_target][data_type_to_number]=-1

		final_energy += dataset["models"][model_source]["energyCost"]


	for model_id in model_used:
		if model_id in model_used_as_source:
			continue
		model=dataset["models"][model_id]
		for data_type,data_req in model["dataRequirements"].items():
			val = model_fills[model_id][DATA_TYPE_LIST.find(data_type)]
			if val == -1:
				continue
			if val<data_req["lowerBound"]:
				return 0, False, f"Model {model_id} has too few data of type {data_type}"
			if val>data_req["upperBound"]:
				return 0, False, f"Model {model_id} has too many data of type {data_type}"

		if model_defiled[model_id]:
			final_value += model["value"]//2
		else:
			final_value += model["value"]

		final_energy += model["energyCost"]

	if final_energy>dataset["energyCap"]:
		return 0, False, f"The total energy is too high ({final_energy}/{dataset['energyCap']})"

	if final_value and show_details:
		print(f"\nTotal value: {final_value}")
		print(f"Energy used: {final_energy} / {dataset['energyCap']} "
			f"({100*final_energy/dataset['energyCap']:.2f}%)")
		number_of_model_used,total_model=len(model_used),len(dataset['models'])
		print(f"Number of used models: {number_of_model_used} / {total_model} "
			f"({100*number_of_model_used/total_model:.2f}%)")
		number_of_devalued_models=list(model_defiled.values()).count(1)
		print(f"  Number of devalued models: {number_of_devalued_models} "
			f"({100*number_of_devalued_models/number_of_model_used:.2f}%)")
		number_of_data_sources=len(model_used_as_source)
		print(f"  Number of models used as data source: {number_of_data_sources} "
			f"({100*number_of_data_sources/number_of_model_used:.2f}%)")
		data_used,total_data=len(solution['dataMappings']),len(dataset['datasets'])
		print(f"Number of used datasets: {data_used} / {total_data} "
			f"({100*data_used/total_data:.2f}%)")

	return final_value, True, ""



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-d","--dataset",
		help="Path to the dataset file",default="datasets/1_example.json")
	parser.add_argument("-s","--solution",
		help="Path to the solution file",default="solutions/1_example_submission.json")
	parser.add_argument("--details", action="store_true",
		help="Show more details on the submitted solution")
	args=parser.parse_args()
	with open(args.solution) as fi:
		solution = fi.read()
	with open(args.dataset) as fi:
		dataset = fi.read()
	score, is_valid, message = getSolutionScore(solution, dataset,show_details=args.details)

	if is_valid:
		print('✅ Solution is valid!')
		print(f'Score: {score:_}')
	else:
		print('❌ Solution is invalid')
		print(f'{message}')