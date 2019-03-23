#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 17:10:45 2018

@author: mwnthai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import tree,ensemble,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

metrics=pd.DataFrame(index=['accuracy','precision','recall'],
                     columns=['NULL','decisiontree','randomforest'])
#from rule import Rule
#from rule_extraction import draw_tree,rules_vote

dataset=pd.read_csv('final_file.csv', usecols=['sex','age','BP','Blood Suger(rbs)','bmi','area','result'])
dataset.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(dataset[['sex','age','BP','Blood Suger(rbs)','bmi','area','result']], 
                                                    dataset.result, test_size=0.2, random_state=0)

# Sex
X_train.groupby(['sex'])['result'].mean()
ordered_labels = X_train.groupby(['sex'])['result'].mean().to_dict()
ordered_labels
# Mean Encoding
X_train['Sex_ordered'] = X_train.sex.map(ordered_labels)
X_test['Sex_ordered'] = X_test.sex.map(ordered_labels)
# area
X_train.groupby(['area'])['result'].mean()
ordered_labels = X_train.groupby(['area'])['result'].mean().to_dict()
ordered_labels
# Mean Encoding
X_train['area_ordered'] = X_train.area.map(ordered_labels)
X_test['area_ordered'] = X_test.area.map(ordered_labels)

print(X_train.shape, X_test.shape)
print(X_train.head(5))

#creating training and test set
X_train_proceeded = X_train[['Sex_ordered', 'age','BP','Blood Suger(rbs)','bmi','area_ordered']]
X_test_proceeded = X_test[['Sex_ordered', 'age','BP','Blood Suger(rbs)','bmi','area_ordered']]
print(X_train_proceeded.head())

#creating model
model_tree_clf = RandomForestClassifier( n_estimators=50, random_state=42)
model_tree_clf.fit(X_train_proceeded,y_train)
#evaluating model
pred=model_tree_clf.predict(X_test_proceeded)
metrics.loc['accuracy','decisiontree']=accuracy_score(y_pred=pred,y_true=y_test)
metrics.loc['precision','decisiontree']=precision_score(y_pred=pred,y_true=y_test)
metrics.loc['recall','decisiontree']=recall_score(y_pred=pred,y_true=y_test)
#confusion metrix
CM=confusion_matrix(y_pred=pred,y_true=y_test)
print(CM)

# model performance on training set
y_pred = model_tree_clf.predict(X_train_proceeded)
confusion=confusion_matrix(y_train,y_pred)
confusion

#creating graph
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
import graphviz
from sklearn import tree

from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_tree_clf.estimators_[4]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = X_train_proceeded.columns, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

#creating features
featurelength=np.arange(len(list(X_train_proceeded)))
names=list(X_train_proceeded)
importances=pd.DataFrame({"Features":list(X_train_proceeded),"Importance":model_tree_clf.feature_importances_})
importances.sort_values(by='Importance',ascending=False,inplace=True)
plt.figure(figsize=(10,5))
plt.title("Feature Importance")
plt.bar(featurelength,importances["Importance"],align="center",color="blue")
plt.xticks(featurelength,importances["Features"],rotation="60")
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
