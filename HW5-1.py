# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:36:05 2022

@author: ramra
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error 

df = pd.read_csv('C:/Data Science and Analytics/CS 5033/Homeworks/train.csv')

print(df.head())

X = df.iloc[:, :-1]

Y = df.iloc[:,-1]

print(len(X))

print(len(Y))

# Exercise 1 a

# splitting train and test into an 80/20 split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#print(X_train.head(), len(X_train))

#print(X_test.head(), len(X_test))

#print(y_train.head(), len(y_train))

#print(y_test.head(), len(y_test))

# Exercise 1 b

# splitting the train dataset into 75/25 split

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size = 0.25)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))




enet = ElasticNet(alpha = 0, l1_ratio = 0, max_iter =10000)
enetfitted = enet.fit(X_train1, y_train1)

print(enetfitted)
    
y_predict = enetfitted.predict(X_train1)
mse = mean_squared_error(y_predict, y_train1)

print(mse)    

