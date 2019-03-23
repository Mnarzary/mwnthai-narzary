#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 00:54:21 2018

@author: mwnthai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('dotma.csv')
x=dataset.iloc[:,[3,5,6]].values
y=dataset.iloc[:,7].values

"""print(dataset.groupby('Cause of Death').size())"""


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])

x[:,1]=labelencoder.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

labelencodery=LabelEncoder()
y=labelencoder.fit_transform(y)
onehotencoder=OneHotEncoder(categorical_features=[0])
y=y.reshape((y.shape[0],1))
y=onehotencoder.fit_transform(y).toarray()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
for K in range(15):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
 neigh.fit(x_train, y_train) 
 y_pred = neigh.predict(x_test)
 print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))

y_pred=knn.predict(x_test)

"""import seaborn as sns
sns.countplot(dataset['Cause of Death'],label="Count")
plt.show()"""

from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
