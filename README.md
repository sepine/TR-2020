# TR-2020

'datas/New_Dataset' is the directory of data files, in which the last column is the label of each instance.

'results/' contains the experimental outputs for the three RQs in terms of two effort-aware indicators and an example for drawing the boxplot.

'data_helper.py' implements the data preprocess that generates the triplets used in this work.

'data_loader.py' implements the data partition that generates the training and test set from original datasets.

'classifiers.py' contains the classifier used in this work.

'CT_Model.py' implements the whole model and saves the results in a specific directory, which also contains the default parameters.

We only need to run the script with 'python CT_Model.py' and the results will save in the directory 'datas/CT_ALL_output'

This code should be run with tensorflow library.
