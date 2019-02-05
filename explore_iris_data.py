#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:10:44 2019

@author: jason
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    DATA_FILE = './data/iris.csv'
    print('Data file location: {}'.format(DATA_FILE))

    print('Reading data file...')
    DATAFRAME = pd.read_csv(DATA_FILE)

    print('Summary information for dataset:')
    print(DATAFRAME.describe())

    print('Accessing features present in dataset:')
    print(DATAFRAME.columns)

    print('Saving species information in data vector')
    CLASS = 'Name'
    Y = DATAFRAME[CLASS]

    print('Saving species feature observations to dataframe')
    FEATURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

    X = DATAFRAME[FEATURES]

    print('Summary information for data features:')
    print(X.describe())
    print('Display first 5 observations from feature dataframe')
    print(X.head())

    # split data into training, validation sets
    # static random state for reproducibility
    print('Split features and labels into training and validation sets')
    TRAIN_X, VAL_X, TRAIN_Y, VAL_Y = train_test_split(X, Y, random_state=1)

    # create classifier model
    print('Create decision tree classification model')
    MODEL = DecisionTreeClassifier(random_state=1)
    MODEL.fit(TRAIN_X, TRAIN_Y)

    print('Use model to predict species names from features in validation set')
    VAL_PREDICTIONS = MODEL.predict(VAL_X)
    MODEL_ACCURACY = accuracy_score(VAL_Y, VAL_PREDICTIONS)
    print('Compare predictions vs validation set labels.')
    print('Accuracy of model using validation data: {}'.format(MODEL_ACCURACY))
