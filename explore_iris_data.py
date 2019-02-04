#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:10:44 2019

@author: ultra_jason
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    iris_data_file = '~/anaconda3/lib/python3.6/site-packages/pandas/tests/data/iris.csv'
    iris_data = pd.read_csv(iris_data_file)
    iris_data.describe()
    
    iris_data.columns
    
    y = iris_data.Name
    
    iris_features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    
    X = iris_data[iris_features]
    
    X.describe()
    X.head()
    
    # split data into training, validation sets
    # static random state for reproducibility
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    
    # create classifier model
    iris_model = DecisionTreeClassifier(random_state=1)
    iris_model.fit(train_X, train_y)
    
    val_predictions = iris_model.predict(val_X)
    print(accuracy_score(val_y, val_predictions))