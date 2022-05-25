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

df = pd.read_csv('train.csv')

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


lambda1 = [0, 1E-6, 1E-4, 1E-3, 1E-2, 1E-1, 1]
lambda2 = [0, 1E-6, 1E-4, 1E-3, 1E-2, 1E-1, 1]

alphalist = []

l1_ratiolist = []

enetlist =[]

for i in lambda1:
    for j in lambda2:
        if i+j <=1 and  (i+j) !=0:
            alpha = i +j 
            l1_ratio = i/(i+j)
            alphalist.append(alpha)
            l1_ratiolist.append(l1_ratio)

print(alphalist)
print(l1_ratiolist)

for i in range(len(alphalist)):
    enet = ElasticNet(alpha = alphalist[i], l1_ratio = l1_ratiolist[i])
    enetfitted = enet.fit(X_train1, y_train1)
    enetlist.append(enetfitted)

print(enetlist)

predictions =[]
rmse = []

for i in range(len(enetlist)):
    y_predict = enetlist[i].predict(X_train1)
    mse = mean_squared_error(y_predict, y_train1)
    predictions.append(y_predict)    
    rmse.append(mse)
    
print(len(predictions))
print(predictions[0])
print(rmse)

