# IA responsables

Un centre de calcul vient d'être construit pour entraîner un grand nombre de modèles d'intelligence artificielle. Ces modèles nécessitent toutefois une quantité importante d'énergie pour tous être testés et entraînés. Or, pour limiter son impact environnemental, le centre a décidé de fixer une limite d’énergie.

Vous avez été engagé pour déterminer quels modèles entraîner parmi tous ceux à disposition et lesquels laisser de côté. Pour vous aider, vos ingénieurs ont préalablement étudié les modèles possibles et ont déterminé pour chacun une valeur représentant la plus-value qu'apporterait leur développement. De plus, étant donné la variété des cadres d'utilisation de chacun des modèles: analyse d'images; systèmes de recommandation; détection de fraudes, ces derniers ont également des besoins en données d'entraînement qui diffèrent de l'un à l'autre.
Quatre type de données on été catégorisées:
- `n`: des données numériques,
- `t`: des données textuelles en langage naturel,
- `i`: des images,
- `c`: des extraits de code.

À chacun de ces besoins est associé un intervalle définissant le montant de données nécessaire à l’entraînement optimal du modèle. L'intervalle comprend un montant minimum en dessous duquel le modèle n'est pas suffisamment abouti ainsi qu'un montant maximum au delà duquel les nouvelles données ingérées cessent de l'améliorer significativement, constituant ainsi un gâchis de puissance de calcul.

Vous avez également à disposition une banque de datasets définis par leur type de données, leur taille ainsi qu'une valeur déterminant si les données sont libres de droit.
Afin d'éviter au maximum les possibles biais d'apprentissage, chaque dataset ne peut être utilisé que pour l'entraînement d'un modèle.

Enfin, il est également possible d'utiliser certains modèles comme générateur de données pour satisfaire entièrement l'un des besoins d'un autre modèle. Toutefois, seuls les modèles ayant un unique besoin en données peuvent être utilisés de cette manière. Dans un tel cas, seul l'unique type du modèle source sera satisfait pour le modèle cible. Si le modèle cible possède d'autres besoin en données, ils devront être satisfait soit par d'autres datasets soit par d'autres modèles dont le type correspond. Pour éviter un effondrement général des performances, un modèle ayant utilisé un autre comme source de données ne peut à son tour être utilisé comme source de données.

Votre rôle est donc de déterminer les modèles à entraîner et d'affecter à chaque le bon nombre de dataset ou de modèle de manière à satisfaire leurs besoin en données.

## Structure des datasets

```json
{
  "comment": "...", // Un commentaire spécifique au jeu de données
  "energyCap": 156489456, // Le montant d'énergie à ne pas dépasser
  "models": [
    {
      "id": 0, // L'identifiant du modèle
      "value": 750, // La valeur métier du modèle
      "energyCost": 4198831, //Le coût en énergie
      "dataRequirements": {
        "i": { // Le type de données d'un des besoins
          "lowerBound": 68384, // Le montant minimum de données dont le modèle a besoin
          "upperBound": 72636  // Le montant maximum de données nécessaire au modèle
        }
      }
    },
    {
      "id": 1,
      "value": 503,
      "energyCost": 6184831,
      "dataRequirements": {
        "t": {
          "lowerBound": 28410,
          "upperBound": 31690
        },
        "i": {
          "lowerBound": 26279,
          "upperBound": 29385
        }
      }
    },
    ...
  ],
  "datasets": [
  	{
  	  "id": 0, // L'identifiant d'un dataset
  	  "dataType": "c", // Le type de données du dataset
  	  "size": 1648, // La taille du dataset
  	  "isCopyrighted": 1 // 0 si les données sont libres de droits, 1 sinon
  	},
  	{
  	  "id": 1,
  	  "dataType": "n",
  	  "size": 9943,
  	  "isCopyrighted": 0
  	},
  	...
  ]
}
```

## Structure de la solution attendue

```json
{
  "dataMappings": [
  	[1,6], // Affecte le dataset 1 au modèle 6
    [5,3],
  	[7,3],
  	...
  ],
  "modelMappings": [
  	[4,9], // Affecte le modèle 4 comme source de données au modèle 9
  	[15,0],
  	...
  ]
}
```

## Calcul du score

Le score de votre solution est égal à la **somme des valeurs** des modèles entraînés.

Pour des raisons éthiques, les modèles utilisant au moins un dataset non libre de droits verront leur valeur divisée par deux arrondie à l'inférieur.
De la même manière, ceux utilisant des données générées auront également leur valeur divisée par deux.
Les deux malus ne peuvent toutefois par se cumuler.
Enfin, la valeur des modèles utilisés comme source de données ne sera pas prise en compte, seul leur coût en énergie sera comptabilisé.

Si votre solution ne respecte pas les contraintes (pas assez de données, dataset attribué plusieurs fois, problèmes de types...), elle sera considérée comme invalide et ne rapportera aucun point.

---

## Classement global

Pour chaque jeu de données, le joueur avec la meilleure solution (valeur totale la plus haute) remporte 1 million de points de classement. Le score des autres joueurs est calculé en fonction de la solution du meilleur joueur avec la formule suivante :

```
1 000 000 * score_meilleur / score_joueur
```

Par exemple, si votre solution coûte deux fois plus cher que celle du meilleur adversaire, votre score sur ce jeu de données sera de 500 000 points.

Le classement est déterminé en fonction du total de points obtenus. Vous pouvez soumettre autant de fois que vous le souhaitez, seule votre meilleure soumission sur chaque jeu de données sera prise en compte.

## Scripts d'aide

La présente archive fournit, en plus des fichiers d'instance, d'un exemple de fichier de solution et des énoncés, deux scripts d'aide:
- Le script `starter.py` fournit une implémentation naïve du problème, utilisable comme point de départ de votre algorithme de résolution.
- Le script `test_solutions.py` permet de calculer le score de l'une de vos solutions. Il détermine également si la solution proposée est valide ou non. Vous pouvez utiliser ce script en tant que script autonome de la façon suivante:
```
python test_solution.py [-h] 
[-d DATASET] [-s SOLUTION] [--details]}

options:
  -h, --help    affiche ce message
  -d DATASET, --dataset DATASET
                Chemin vers une fichier d'entrée
  -s SOLUTION, --solution SOLUTION
                Chemin vers un fichier de solution
  --details     Affiche des statistiques sur la solution donnée
```
