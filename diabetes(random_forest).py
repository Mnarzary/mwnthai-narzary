#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:49:03 2018

@author: mwnthai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:42:13 2018

@author: mwnthai
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('final_file.csv')

cols_to_std = ['sex','age','BP','Blood Suger(rbs)','bmi',
                'area']

x=dataset.iloc[:,[0,1,2,3,4,5,6]].values
y=dataset.iloc[:,7].values
z=dataset.iloc[:,[2,3,4,5,7]]
"""print(dataset.groupby('Cause of Death').size())"""


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode=LabelEncoder()
onehotencoder=OneHotEncoder(categorical_features=[0])
x[:,1]=encode.fit_transform(x[:,1])
x[:,6]=encode.fit_transform(x[:,6])
x=onehotencoder.fit_transform(x).toarray()
cols_to_std = ['age','BP','Blood Suger(rbs)','bmi','result'
                ]
z[cols_to_std] = z[cols_to_std].apply(lambda x: (x-x.mean()) / x.std())
feature_list=cols_to_std

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=250,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)

from sklearn.ensemble import RandomForestClassifier
#finding accuracy
rf1 = RandomForestClassifier( n_estimators=100, random_state=42)
rf1.fit(x_train, y_train)
y_pred=rf1.predict(x_test)
print(y_pred)
errors = abs(y_pred - y_test)
print(errors)
mape = 100 * (errors / y_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix=confusion_matrix(y_test,y_pred)
accuracy_score=accuracy_score(y_test,y_pred)
confusion_matrix
accuracy_score


from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf1.estimators_[6]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

r=RandomForestClassifier()
r.fit(x_train, y_train)
pr=r.predict(x_test)
print("Accuracy of RandomForest on training set: {:.3f}".format(rf1.score(x_train, y_train)))
print("Accuracy of RandomForest on test set: {:.3f}".format(rf1.score(x_test, y_test)))
print('classification_report')
print(classification_report(y_test,pr))
print('confusion_metrix')
print(confusion_matrix(y_test,pr))
