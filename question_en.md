# Responsible AI

A data centre just opened to train a large number of artificial intelligence models. However, these models require a significant amount of energy to be tested and trained. To minimise its environmental impact, the centre has decided to set an energy limit.

You have been hired to determine which models to train from among all those available, and which to leave out. To assist you, your engineers have already studied the possible models and determined, for each one, the added value that their development would bring. Furthermore, given the variety of use cases for each model: image analysis, recommendation systems, fraud detection and more, they also have different training data requirements.
Four types of data have been categorised:
- `n`: numerical data,
- `t`: natural language text data,
- `i`: images,
- `c`: code snippets.

Each of these requirements is associated with a range defining the amount of data required for optimal model training. The range includes a minimum amount, below which the model is not sufficiently refined, and a maximum amount, beyond which any new data ingested ceases to improve it significantly, thereby constituting a waste of computing power.

You also have access to a repository of datasets categorised by their data type, their size and a flag indicating whether the data is free to use.
To minimise training bias as much as possible, each dataset may only be used to train a single model.

Finally, it is also possible to use certain models as data generators to fully satisfy one of the requirements of another model. However, only models with a single data requirement can be used in this way. In such cases, only the single data type of the source model will be satisfied for the target model. If the target model has other data requirements, these must be satisfied either by other datasets or by other models of the corresponding type. To prevent a general performance collapse, a model that has used another model as a data source cannot itself be used as one.

Your role is therefore to determine which models to train and to assign the correct number of datasets or models to each one in order to satisfy their data requirements.


## Datasets structure

```json
{
  "comment": "...", // A comment specific to the dataset
  "energyCap": 156489456, // The maximum amount of energy consumable
  "models": [
    {
      "id": 0, // The model identifier
      "value": 750, // The added value of the model
      "energyCost": 4198831, //The energy consumption of the model
      "dataRequirements": {
        "i": { // The data type of one of the requirements
          "lowerBound": 68384, // The minimum amount of data needed by the model
          "upperBound": 72636  // The maximum amount of data needed by the model
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
  	  "id": 0, // The dataset identifier
  	  "dataType": "c", // The dataset data type
  	  "size": 1648, // The dataset size
  	  "isCopyrighted": 1 // 0 if the dataset is free to use, 1 otherwise
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

## Structure of the solutions

```json
{
  "dataMappings": [
  	[1,6], // Assigns dataset 1 to model 6
    [5,3],
  	[7,3],
  	...
  ],
  "modelMappings": [
  	[4,9], // Assigns model 4 as a data source to model 9
  	[15,0],
  	...
  ]
}
```

## Score computation

Your solution’s score is equal to the **sum of the values** of all the models that you trained.

For ethical reasons, models using at least one dataset that is not free to use will have their value halved and rounded down.
Similarly, those using generated data will also have their value halved.
However, these two penalties cannot be combined.
Finally, the value of models used as data sources will not be taken into account; only their energy cost will be counted.

If your solution does not meet the constraints (insufficient data, a dataset assigned more than once, type mismatch, etc.), it will be considered invalid and will not earn any points.

---

## Classement global

For each dataset, the player with the best solution wins 1 million points. The scores of the other players are calculated relative to the best player’s solution using the following formula:

```
1,000,000 * best_score / player_score
```

For example, if your solution costs twice as much as that of the best opponent, your score for that dataset will be 500,000 points.

The ranking is determined by the total number of points scored. You may submit as many times as you like; only your best submission for each dataset will be taken into account.

## Helper scripts

In addition to the instance files, a sample solution file and the problem statements, this archive provides two helper scripts:
- The `starter.py` script provides a naive implementation of the problem, which can be used as a starting point for your solution algorithm.
- The `test_solutions.py` script allows you to calculate the score of one of your solutions. It also determines whether the proposed solution is valid or not. You can use this script as a standalone script as follows:
```
python test_solution.py [-h] 
[-d DATASET] [-s SOLUTION] [--details]}

options:
  -h, --help    show this help message and exit
  -d DATASET, --dataset DATASET
                Path to the dataset file
  -s SOLUTION, --solution SOLUTION
                Path to the solution file
  --details     Show statistics on the solution
```
