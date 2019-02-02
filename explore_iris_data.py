#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 18:10:44 2019

@author: ultra_jason
"""

import pandas as pd

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
    