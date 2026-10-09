# Responsible AI – solveur

Code pour le challenge « Responsible AI » (starter kit Euro-Information 2026).

## Approche

1. **Sélection par MIP** (`select_mip.py`, HiGHS via `scipy.optimize.milp`).
   Chaque modèle peut être :
   - `full` : tous ses besoins sont couverts par des datasets libres (valeur `v`) ;
   - `half` : datasets quelconques et/ou modèles sources (valeur `v // 2`) ;
   - `source` : modèle mono-type utilisé comme générateur (valeur 0, énergie comptée).
   Les données sont agrégées par type (consommation = `lowerBound`). On ajoute aussi
   des contraintes d'énergie et de nombre de sources disponibles par type. Le
   paramètre `alpha` garde une petite marge sur les données pour absorber le gaspillage.
2. **Affectation des datasets** (`assign.py`) : type par type, chaque besoin
   `[lb, ub]` est rempli d'abord avec de gros datasets, puis on termine par un
   subset-sum (bitset DP) qui vise un gaspillage minimal. Les besoins des modèles `full`
   n'utilisent que des datasets libres ; les autres utilisent d'abord des datasets
   sous copyright. Plusieurs ordres de traitement sont essayés.
3. **Réparation** (`solve.py`) : les modèles qu'on n'a pas pu remplir sont retirés,
   les sources sont appariées aux besoins, puis un glouton ajoute des modèles avec
   l'énergie et les données restantes. La solution est vérifiée avec
   `test_solution.py` et n'est écrite dans `solutions/` que si elle bat celle déjà présente.

## Utilisation

```
pip install highspy scipy numpy sortedcontainers
python solve.py datasets/2_medium.json [temps_limite_mip_s] [alpha]
python test_solution.py -d datasets/2_medium.json -s solutions/2_medium_submission.json
```
