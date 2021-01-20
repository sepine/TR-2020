# TR-2020

## Contents

````datas/dataset```` is the directory of data files, in which the last column is the label of each instance.

````results/```` contains the experimental outputs for the three RQs in terms of two effort-aware indicators and an example for drawing the boxplot.

````data_helper.py```` implements the data preprocess that generates the triplets used in this work.

````data_loader.py```` implements the data partition that generates the training and test set from original datasets.

````classifiers.py```` contains the classifier used in this work.

````indicator.py```` implements the used two effort-aware indicators.

````model.py```` implements the whole model and saves the results in a specific directory, which also contains the default parameters.

## Dependencies
* Python 3.6
* Tensorflow 1.14.0
* Scipy
* Numpy
* Scikit-learn

## Run

````python model.py```` 

The results will save in the directory ````results/````
