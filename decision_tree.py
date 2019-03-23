#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:03:04 2018

@author: mwnthai
"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier #Decision Tree
import matplotlib.pyplot as plt # To plot graphs
from sklearn.metrics import accuracy_score # To test accuracy
from sklearn import tree
import pydot

churn=pd.read_csv('final_file.csv')
trainsize=0.8
indx=np.random.rand(len(churn))<trainsize
train_churn=churn[indx]
test_churn=churn[~indx]

features_train=train_churn.iloc[:,[1,2,3,4,5,6]].values
target_train=train_churn.iloc[:,7].values
print(list(features_train))
"""print(dataset.groupby('Cause of Death').size())"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode=LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])
features_train[:,0]=encode.fit_transform(features_train[:,0])

features_train[:,5]=encode.fit_transform(features_train[:,5])
features_train=onehotencoder.fit_transform(features_train).toarray()


dtree=DecisionTreeClassifier(criterion="entropy")
trained=dtree.fit(features_train,target_train)
#ploting feature graph
featurelength=np.arange(len(list(features_train)))
names=list(features_train)
a={"Features":features_train,"Importance":trained.feature_importances_}
importances=pd.DataFrame.from_dict(a,orient='index')

importances.sort_values(by="Importance",ascending=True,inplace=False)
plt.figure(figsize=(10,5))
plt.title("Feature Importance")

plt.bar(featurelength,importances["Importance"],align="center",color="blue")
plt.xticks(featurelength,importances["Features"],rotation="60")
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# Splitting test dataset into target and features
features_test=test_churn.iloc[:,[1,2,3,4,5,6]].values
target_test=test_churn.iloc[:,7].values

onehotencoder = OneHotEncoder(categorical_features = [0])
features_test[:,0]=encode.fit_transform(features_test[:,0])

features_test[:,5]=encode.fit_transform(features_test[:,5])
features_test=onehotencoder.fit_transform(features_test).toarray()

#  Predicting target for train and test dataset
results_test=trained.predict(features_test)
results_train=trained.predict(features_train)

test_accuracy=accuracy_score(target_test,results_test)
train_accuracy=accuracy_score(target_train,results_train)
print("Accuracy for the test dataset is %.3f%% and accuracy for the training dataset is %.3f%%" %(test_accuracy*100,train_accuracy*100))

#ploting tree


splits=np.arange(10,1000,100)
leafnodes=np.arange(10,1000,100)

def DtreeIter(train_features,train_target,test_features,test_target,samplesplit,sampleleaf):
    treeOpt=DecisionTreeClassifier(criterion="entropy",min_samples_split=samplesplit,min_samples_leaf=sampleleaf)
    treeOpt=treeOpt.fit(train_features,train_target)
    result_Opt=treeOpt.predict(test_features)
    return accuracy_score(test_target,result_Opt)

result_optimise=dict()
for values in splits:
    result_optimise[values]=list()

for values in splits:
    for nodes in leafnodes:
        result_optimise[values].append([DtreeIter(features_train,target_train,features_test,target_test,values,nodes)])

optimal_split=max(result_optimise, key=lambda x: result_optimise[x][1])
optimal_accuracy=max(result_optimise[optimal_split])
optimal_leaf=leafnodes[list(result_optimise[optimal_split]).index(optimal_accuracy)]
print("Optimal 'Sample Split Size' is %d and 'Optimal Leaf Samples' are %d. Best accuracy is %.2f%%" %(optimal_split,optimal_leaf,optimal_accuracy[0]*100))

plt.figure(figsize=(10,5))
plt.plot(splits,result_optimise[leafnodes[0]],'b',label='Leaf={}'.format(leafnodes[0]))
plt.plot(splits,result_optimise[leafnodes[1]],'r',label='Leaf={}'.format(leafnodes[1]))
plt.plot(splits,result_optimise[leafnodes[2]],'y',label='Leaf={}'.format(leafnodes[2]))
plt.plot(splits,result_optimise[leafnodes[3]],'g',label='Leaf={}'.format(leafnodes[3]))
plt.plot(splits,result_optimise[leafnodes[4]],'c',label='Leaf={}'.format(leafnodes[4]))
plt.plot(splits,result_optimise[leafnodes[5]],'m',label='Leaf={}'.format(leafnodes[5]))
plt.plot(splits,result_optimise[leafnodes[6]],'k',label='Leaf={}'.format(leafnodes[6]))
plt.plot(splits,result_optimise[leafnodes[7]],'b',label='Leaf={}'.format(leafnodes[7]))
plt.plot(splits,result_optimise[leafnodes[8]],'r',label='Leaf={}'.format(leafnodes[8]))
plt.plot(splits,result_optimise[leafnodes[9]],'y',label='Leaf={}'.format(leafnodes[9]))
plt.legend(loc=4)
plt.xlabel('Min Sample Splits')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy')
plt.show()
plt.savefig('decision_tree')




churn.head()

plt.figure(figsize = (11,6))
sns.countplot(x=churn['area'], hue=churn['result'], palette = 'Set1')

churn.info()