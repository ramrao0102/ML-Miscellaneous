# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:13:25 2022

@author: ramra
"""

# HW6 for CS5033

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import math
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

df = pd.read_csv('C:/Data Science and Analytics/CS 5033/Homeworks/Data_for_UCI_named.csv')

df.drop(columns=['p1'], inplace=True)
df.drop(columns=['stab'], inplace=True)

print(df.head())

# Exercise 1

for i in df.index:
    if df['stabf'][i] == 'stable':
        df['stabf'][i] = 1
    else:
        df['stabf'][i] = 0
        
print(df.dtypes)
print(df.head())

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

print(X_train.head(), len(X_train))

print(X_test.head(), len(X_test))

print(y_train.head(), len(y_train))

print(y_test.head(), len(y_test))

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.25)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))

# Exercise 2, 
# Neural Nets, MLP Classifier

# Model 1 , 1 Hidden Layer of 20 Units

clf = MLPClassifier(random_state=1, hidden_layer_sizes=(20,), max_iter=300).fit(X_train1, y_train1)

probability = clf.predict_proba(X_train1)

cr_entropy = []

actual = []

y_tr1 = y_train1.to_numpy()

for i in range(len(y_train1)):
    a = [0 for j in range(2)]
    a[y_tr1[i]] = 1
    actual.append(a)

for i in range(len(probability)):
    cross_entropy =0.0
    for j in range(len(probability[i])):
        if probability[i][j] ==0:
            probability[i][j] = probability[i][j] + 1e-15
        elif probability[i][j] ==1:
            probability[i][j] = probability[i][j] - 1e-15 
        else:
            probability[i][j] = probability[i][j]
        cross_entropy += - actual[i][j]*math.log(probability[i][j]) 
    
    cr_entropy.append(cross_entropy)

sum =0.0
for i  in range(len(y_tr1)):
    sum += cr_entropy[i]

mean = sum/len(y_tr1)

print("Mean Cross Entropy on Train Dataset, Model 1")
print(mean)
print("__________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Train Dataset, Model 1")

print(log_loss(y_tr1, probability))

print("__________________________________________________________________")


# Model 2, 2 Hiden Layers with 10 Units Each

clf1 = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=300).fit(X_train1, y_train1)

probability = clf1.predict_proba(X_train1)

cr_entropy1 = []

for i in range(len(probability)):
    cross_entropy =0.0
    for j in range(len(probability[i])):
        if probability[i][j] ==0:
            probability[i][j] = probability[i][j] + 1e-15
        elif probability[i][j] ==1:
            probability[i][j] = probability[i][j] - 1e-15 
        else:
            probability[i][j] = probability[i][j]
        cross_entropy += - actual[i][j]*math.log(probability[i][j])
    
    cr_entropy1.append(cross_entropy)

sum =0.0
for i  in range(len(y_tr1)):
    sum += cr_entropy1[i]

mean = sum/len(y_tr1)

print("Mean Cross Entropy on Train Dataset, Model 2")
print(mean)

print("____________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Train Dataset, Model 1")

print(log_loss(y_tr1, probability))

print("____________________________________________________________________")

y_val11 = y_val1.to_numpy()

# Model 1 on Validation Examples, Hidden Layer with 20 Units

probability1 = clf.predict_proba(X_val1)

cr_entropy = []

actual1 = []

for i in range(len(y_val1)):
    a = [0 for j in range(2)]
    a[y_val11[i]] = 1
    actual1.append(a)

probability11 = np.zeros((len(y_val1) ,2)) #Initialize Array

for i in range(len(probability1)):
    cross_entropy =0.0
    for j in range(len(probability1[i])):
        if probability1[i][j] ==0:
            probability11[i][j] = probability1[i][j] + 1e-15
        elif probability11[i][j] ==1:
            probability11[i][j] = probability1[i][j] - 1e-15 
        else:
            probability11[i][j] = probability1[i][j]
        cross_entropy += - actual1[i][j]*math.log(probability11[i][j]) 
    
    cr_entropy.append(cross_entropy)

sum =0.0
for i  in range(len(y_val11)):
    sum += cr_entropy[i]

mean = sum/len(cr_entropy)

print("Mean Cross Entropy on Validation Dataset, Model 1")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Validation Dataset, Model 1")

print(log_loss(y_val11, probability1))

print("______________________________________________________________________ ")

# Model 2 on Validation Examples

probability2 = clf1.predict_proba(X_val1)

cr_entropy1 = []

probability22 = np.zeros((len(y_val1) ,2)) #Initialize Array

for i in range(len(probability2)):
    cross_entropy =0.0
    for j in range(len(probability2[i])):
        if probability2[i][j] ==0:
            probability22[i][j] = probability2[i][j] + 1e-15
        elif probability22[i][j] ==1:
            probability22[i][j] = probability2[i][j] - 1e-15 
        else:
            probability22[i][j] = probability2[i][j]
        cross_entropy += - actual1[i][j]*math.log(probability22[i][j])
    
    cr_entropy1.append(cross_entropy)

sum =0.0
for i  in range(len(cr_entropy1)):
    sum += cr_entropy1[i]

mean = sum/len(y_val11)

print("Mean Cross Entropy on Validation Dataset, Model 2")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Validation Dataset, Model 2")

print(log_loss(y_val11, probability2))

print("_______________________________________________________________________")

# Model 1 outperforms Model 2 on Cross Entropy Loss
# So Model 1 is to Train on Train+Validation Data

clf3 = MLPClassifier(random_state=1, hidden_layer_sizes=(20,), max_iter=300).fit(X_train, y_train)

probability3 = clf3.predict_proba(X_test)

cr_entropy = []

actual2 = []

y_test1 = y_test.to_numpy()

for i in range(len(y_test)):
    a = [0 for j in range(2)]
    a[y_test1[i]] = 1
    actual2.append(a)

probability33 = np.zeros((len(y_test) ,2)) #Initialize Array

for i in range(len(probability3)):
   cross_entropy =0.0
   for j in range(len(probability3[i])):
       if probability3[i][j] ==0:
           probability33[i][j] = probability3[i][j] + 1e-15
       elif probability33[i][j] ==1:
           probability33[i][j] = probability3[i][j] - 1e-15 
       else:
           probability33[i][j] = probability3[i][j]
       cross_entropy += - actual2[i][j]*math.log(probability33[i][j])
       
   cr_entropy.append(cross_entropy) 

sum =0.0
for i  in range(len(cr_entropy)):
    sum += cr_entropy[i]

mean = sum/len(y_test1)

print("Mean Cross Entropy on Test Dataset, Model 1")

print(mean)

print("________________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Test Dataset, Model 1")

print(log_loss(y_test1, probability3))

print("________________________________________________________________________")

# Model 1 Prediction of Labels Based on Probability Thresholds

predictions1 = []
for i in range(0,1001,1):
    k = float(i/1000)
    label1 = []
    for j in range(len(probability1)):
        
        if probability1[j][1] > k:
            label1.append(1)
        else:
            label1.append(0)

    predictions1.append(label1)
   
  
# Model 2 Prediction of Labels Based on Probability Thresholds

predictions2 = []

for i in range(0,1001,1):
    k = float(i/1000)
    label2 = []
    for j in range(len(probability2)):
        
        if probability2[j][1] > k:
            label2.append(1)
        else:
            label2.append(0)

    predictions2.append(label2)
    
#print(len(predictions2)) 

#print(predictions2[0])

#print(predictions2[1])   
    

# Model 1 for Train+ Validation Dataset

predictions3 = []

for i in range(0,1001,1):
    k = float(i/1000)
    label3 = []
    for j in range(len(probability3)):
        
        if probability3[j][1] > k:
            label3.append(1)
        else:
            label3.append(0)

    predictions3.append(label3)
    
print(len(predictions3[0]))

print(len(y_val1))

# Random Predictor

RX = []
RY = []

for i in range(0,1001, 1):
    k = float(i/1000)
        
    RX.append(k)
    RY.append(k)


# compute True Positive Rate and True Negative Rate 

# Model 1 Prediction of Labels Based on Probability Thresholds on Validation Dataset

TPR1 = []
FPR1 = []
Youden_Index1 = 0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions1[i])):
        if predictions1[i][j] ==0:
            if y_val11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions1[i][j] == 1:
            if y_val11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate1 = tp/(tp+fn)
    else:
        true_positive_rate1 =0
    
    
    
    if (tn +fp) !=0:
        false_positive_rate1 = fp/(tn+fp)
    else:
        false_positive_rate1 = 0
    
    if (true_positive_rate1 - false_positive_rate1) > Youden_Index1:
        Youden_Index1 = (true_positive_rate1 - false_positive_rate1)
        final_index = k
       
    
    
    TPR1.append(true_positive_rate1)
    FPR1.append(false_positive_rate1)

print("Probability Threshold with Highest Index", final_index)        
plt.plot(FPR1, TPR1)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["ANN Model 1 Validation Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 1 Validation Dataset is:")
print(Youden_Index1)
        
# Model 2 Prediction of Labels Based on Probability Thresholds on Validation Dataset

TPR2 = []
FPR2 = []
Youden_Index2 = 0.0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions2[i])):
        if predictions2[i][j] ==0:
            if y_val11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions2[i][j] == 1:
            if y_val11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate2 = tp/(tp+fn)
    else:
        true_positive_rate2 =0
    
   
    
    if (tn +fp) !=0:
        false_positive_rate2 = fp/(tn+fp)
    else:
        false_positive_rate2 = 0
          
    if (true_positive_rate2 - false_positive_rate2) > Youden_Index2:
       Youden_Index2 = (true_positive_rate2 - false_positive_rate2)
       final_index = k
    
    TPR2.append(true_positive_rate2)
    FPR2.append(false_positive_rate2)

print("Probability Threshold with Highest Index", final_index)             
plt.plot(FPR2, TPR2)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["ANN Model 2 Validation Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 2 Validation Dataset is:")
print(Youden_Index2)

# Model 1 Prediction of Labels Based on Probability Thresholds for Train + Validation Dataset

TPR3 = []
FPR3 = []
Youden_Index3 = 0.0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions3[i])):
        if predictions3[i][j] ==0:
            if y_test1[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions3[i][j] == 1:
            if y_test1[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate3 = tp/(tp+fn)
    else:
        true_positive_rate3 =0
    
   
    
    if (tn +fp) !=0:
        false_positive_rate3 = fp/(tn+fp)
    else:
        false_positive_rate3 = 0
    
    if (true_positive_rate3 - false_positive_rate3) > Youden_Index3:
      Youden_Index3 = (true_positive_rate3 - false_positive_rate3)
      final_index = k
    
    TPR3.append(true_positive_rate3)
    FPR3.append(false_positive_rate3)

print("Probability Threshold with Highest Index", final_index)         
plt.plot(FPR3, TPR3)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["ANN Model 1 Test Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 1 Test Dataset is:")
print(Youden_Index3)

# Area Under Curve Determinations

# for Model 1, Validation Dataset

AUC1 = 0.0

for i in range(len(FPR1)):

    if i == 0:
        
        prev_coordinate = FPR1[i]
        AUC1 =0.0

    if i >0:
    
        AUC1 += (1/2)* (TPR1[i] +TPR1[i])*(-FPR1[i] + prev_coordinate)   


    prev_coordinate = FPR1[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 1 Validation Dataset")
    
print(AUC1)

print("_____________________________________________________________________________")

print("AUC from In Built Function, Model 1 Validation Dataset")

auc = roc_auc_score(y_val11, probability1[:,1])

print(auc)

print("_____________________________________________________________________________")

# for Model 2, VaLidation Dataset

AUC2 = 0.0

for i in range(len(FPR2)):

    if i == 0:
        
        prev_coordinate = FPR2[i]
        AUC2 =0.0

    if i >0:
    
        AUC2 += (1/2)* (TPR2[i] +TPR2[i])*(-FPR2[i] + prev_coordinate)   

    prev_coordinate = FPR2[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 2 Validation Dataset")
    
print(AUC2)

print("____________________________________________________________________________")


print("AUC from In Built Function, Model 2 Validation Dataset")

auc = roc_auc_score(y_val11, probability2[:,1])

print(auc)

print("_____________________________________________________________________________")


# for Model 1 Prediction on Train + Validation Datasets

AUC3 = 0.0

for i in range(len(FPR3)):

    if i == 0:
        
        prev_coordinate = FPR3[i]
        AUC3 =0.0

    if i >0:
    
        AUC3 += (1/2)* (TPR3[i] +TPR3[i])*(-FPR3[i] + prev_coordinate)   


    prev_coordinate = FPR3[i]
    
print("AUC from Fuction Built from Raw FPR and TPR Data, Model 1 Test Dataset")

print(AUC3)

print("____________________________________________________________________________")

print("AUC from In Built Function, Model 1 Test Dataset")
 
auc = roc_auc_score(y_test1, probability3[:,1])

print(auc)

print("_____________________________________________________________________________")
