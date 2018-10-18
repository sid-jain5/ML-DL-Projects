# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:30:45 2018

@author: siddh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:31:40 2018

@author: siddhant
"""
#K-nearest neighbours (K-NN)

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#Splitting Train/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42)

#Splitting train set into validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.1, random_state = 84)

print("training data points: {}".format(len(y_train)))
print("validation data points: {}".format(len(y_valid)))
print("testing data points: {}".format(len(y_test)))

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting K-NN to dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p=2)
classifier.fit(X_train, y_train)

#Prediting the test set result
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

import math
math.ceil((sum/10500)*100)



