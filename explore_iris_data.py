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
    iris_data_file = './data/iris.csv'
    print('Data file location: {}'.format(iris_data_file))
    
    print('Reading data file...')
    iris_data = pd.read_csv(iris_data_file)
    
    print('Summary information for dataset:')
    print(iris_data.describe())
    
    print('Accessing features present in dataset:')
    print(iris_data.columns)
    
    print('Saving species information in data vector')
    y = iris_data.Name
    
    print('Saving species feature observations to dataframe')
    iris_features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    
    X = iris_data[iris_features]
    
    print('Summary information for data features:')
    print(X.describe())
    print('Display first 5 observations from feature dataframe')
    print(X.head())
    
    # split data into training, validation sets
    # static random state for reproducibility
    print('Split features and labels into training and validation sets')
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    # create classifier model
    print('Create decision tree classification model')
    iris_model = DecisionTreeClassifier(random_state=1)
    iris_model.fit(train_X, train_y)
    
    print('Use model to predict species names from features in validation set')
    val_predictions = iris_model.predict(val_X)
    print('Compare predictions vs validation set labels.')
    print('Accuracy of model using validation data: {}'.format(accuracy_score(val_y, val_predictions)))
    