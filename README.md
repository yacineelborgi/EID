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

## Scripts spécialisés (2e passe)

Constat commun : un besoin avec `lowerBound` minuscule consomme quand même un dataset entier.
`select_mip.py` accepte donc `tiny_lb` (coût exact « k-ième plus petit dataset libre ») et `free_margin`.

- **4_precise** (fenêtres de largeur ~5, aucune source) : `lpinfo.py` (prix duaux LP) →
  `mitm4.py` (meet-in-the-middle autour de l'optimum LP, KD-tree sur énergie + données libres) →
  `pack4.py` (packing exact + ruin & recreate). `mip4.py` = MIP dédiée.
- **6_shortage** : `select_mip2.py` (coût convexe des petits besoins) + `solve2.py`
  (redémarrages avec bruit, réparation par échanges, appariement optimal des sources, ajout glouton).
- **2_medium** : `plan_med.py` + `exact_pack.py` + `med_solve.py`.
- **5_energy / 7_big** : `exact_assign.py`, `energy_iter.py` (MIP corrigée du gaspillage mesuré),
  `big_opt.py` + `rmip.py` (LP → fixation par coûts réduits → MIP restreinte), `eval_plan.py`.

`solve.strict_ok` refuse toute solution avec un id hors bornes ou un modèle utilisé deux fois
(le correcteur fourni ne le vérifie pas).

## Résultats légitimes et plafonds (sans la faille des ids négatifs)

| Fichier | Score | Record du classement | Plafond max (prouvé) | Statut |
|---|---|---|---|---|
| 1_example | 12 946 | 12 946 | 12 946 | optimal prouvé (énumération exhaustive) |
| 2_medium | 272 320 | 272 322 | 272 337 | écart ≤ 17 |
| 3_free | 523 758 | 523 758 | 523 758 | optimal prouvé (Lagrangien exact + énumération) |
| 4_precise | 1 031 941 | 1 031 941 | 1 031 941 | optimal prouvé (`proof_precise/`) |
| 5_energy | 3 786 096 | 3 786 107 | 3 786 149 | écart ≤ 53 |
| 6_shortage | 669 445 | 669 713 | 669 934 | écart ≤ 489 |
| 7_big | 5 822 560 | 5 822 561 | 5 822 563 | écart ≤ 3 |

Les plafonds sont des bornes de dualité faible recalculées en fractions exactes sur une relaxation
valide (coupes « petits besoins » multi-seuils : les k plus petits besoins consomment au moins les
k plus petits datasets). `proofs/` contient des copies de référence des scripts de preuve ; certains
importent des modules de travail qui ne sont pas dans le dépôt.
