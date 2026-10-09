import os
import json
from glob import glob

from test_solution import getSolutionScore

def solve(dataset_txt):
	instance = json.loads(dataset_txt)

	solution = {
		"dataMappings":[],
		"modelMappings":[]
	}

	# Si l'instance est trop grosse: on passe
	if len(instance["models"])*len(instance["datasets"])>1e9:
		print("Instance too big [Passed]")
		return solution

	# Sauvegarde des modèles pouvant servir de source de données	
	useful_mono_models = {data_type:set() for data_type in "ntic"}
	
	data_mappings = {}
	energy_used = 0

	#On parcourt tous les modèles
	for i,model in enumerate(instance["models"],start=1):
		print(end=f"\r{i}/{len(instance['models'])}")

		#Si le modèle est trop coûteux en énergie: on passe
		if energy_used + model["energyCost"]>instance["energyCap"]:
			continue

		data_used = []
		model_used = []
		success = True

		#Pour chacun des types de données dont le modèle a besoin
		for data_type,bounds in model["dataRequirements"].items():

			#S'il existe un modèle en stock pouvant servir de source de données: on l'utilise
			if useful_mono_models[data_type]:
				model_used.append((data_type,useful_mono_models[data_type].pop()))

			data_quantity = 0
			#On passe en revue tous les datasets
			for dataset in instance["datasets"]:
				#Si le dataset a déjà été utilisé pour un précédent modèle: on continue
				if dataset["id"] in data_mappings:
					continue
				#Si le type du dataset ne correspond pas: on continue
				if not dataset["dataType"]==data_type:
					continue
				#Si le dataset est trop volumineux: on continue
				if dataset["size"]+data_quantity>bounds["upperBound"]:
					continue
				#Le dataset est choisi
				data_quantity+=dataset["size"]
				data_used.append(dataset["id"])
				#Si on a suffisamment de données: on stoppe
				if data_quantity>=bounds["lowerBound"]:
					break
			#On vérifie que l'on a le bon nombre de données
			if not bounds["lowerBound"]<=data_quantity<=bounds["upperBound"]:
				success = False
				break

		#Si l'on a pas satisfait l'entièreté des besoins, on remet les modèles utilisés
		#comme source de données dans le stock de ceux disponibles
		if not success:
			for data_type, model_id in model_used:
				useful_mono_models[data_type].add(model_id)
			continue

		energy_used+=model["energyCost"]

		model_id = model["id"]
		#Si le modèle ne nécessitait qu'un seul type de donnée:
		#on le met dans le stock de ceux disponibles comme source de données
		if len(model["dataRequirements"])==1 and not model_used:
			useful_mono_models[data_type].add(model_id)

		#On met à jour les affectations
		data_mappings.update((data,model_id) for data in data_used)
		solution["modelMappings"].extend((model[1],model_id) for model in model_used)

	solution["dataMappings"]=list(data_mappings.items())
	return solution



def main():
	dataset_files = glob("datasets/*.json")

	for filepath in dataset_files:
		filename_input = os.path.basename(filepath)
		filename_output = filename_input.replace(".","_submission.")

		print("\n"+filename_input)
		with open(filepath,"r") as file:
			dataset_txt = file.read()
			solution = solve(dataset_txt)
		
		solution_txt = json.dumps(solution,indent=2)
		
		score, is_valid, message = getSolutionScore(solution_txt,dataset_txt)

		if is_valid:
			print(f"\tSolution saved (Score: {score})")
			with open("solutions/"+filename_output,"w") as file:
				file.write(solution_txt)
		else:
			print("\t"+message)

if __name__ == '__main__':
	main()