#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:33:07 2018

@author: mwnthai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree,ensemble,metrics
from sklearn.metrics import classification_report, precision_recall_curve,precision_score,recall_score,confusion_matrix,accuracy_score

metrics=pd.DataFrame(index=['accuracy','precision','recall'],
                     columns=['NULL','decisiontree','randomforest'])

dataset=pd.read_csv('final_file.csv', usecols=['sex','age','BP','Blood Suger(rbs)','bmi','area','result'])
dataset.dropna(inplace=True)

'''dataset.head()
dataset.info()
print(dataset.groupby('area').size())
import seaborn as sns
sns.countplot(dataset['sex'],label="Count")
plt.show()'''

X_train, X_test, y_train, y_test = train_test_split(dataset[['sex','age','BP','Blood Suger(rbs)','bmi','area','result']], 
                                                    dataset.result, test_size=0.2, random_state=0)

target=y_train
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
#input
area=X_train[['area_ordered']]
variable=X_train[['Sex_ordered','age','BP','Blood Suger(rbs)','bmi','area_ordered']]

#creating model(decisionTreww)
model_tree_clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=5)
model_tree_clf.fit(X_train_proceeded,y_train)
#evaluating model
pred=model_tree_clf.predict(X_test_proceeded)
metrics.loc['accuracy','decisiontree']=accuracy_score(y_pred=pred,y_true=y_test)
metrics.loc['precision','decisiontree']=precision_score(y_pred=pred,y_true=y_test)
metrics.loc['recall','decisiontree']=recall_score(y_pred=pred,y_true=y_test)
#confusion metrix
CM=confusion_matrix(y_pred=pred,y_true=y_test)
print(CM)

#creating model(randomforest)
model_tree_clf1 = RandomForestClassifier( n_estimators=50, random_state=42)
model_tree_clf1.fit(X_train_proceeded,y_train)
#evaluating model
pred=model_tree_clf1.predict(X_test_proceeded)
metrics.loc['accuracy','randomforest']=accuracy_score(y_pred=pred,y_true=y_test)
metrics.loc['precision','randomforest']=precision_score(y_pred=pred,y_true=y_test)
metrics.loc['recall','randomforest']=recall_score(y_pred=pred,y_true=y_test)
#confusion metrix
CM1=confusion_matrix(y_pred=pred,y_true=y_test)
print(CM1)
#evaluation accuracy
100*metrics
#evluation accuracy graph
fig, ax=plt.subplots(figsize=(10,5))
metrics.plot(kind='barh',ax=ax)
ax.grid();

#adjust precision and recall
precision_nb,recall_nb, threshold_nb=precision_recall_curve(y_true=y_test,
                                                            probas_pred=model_tree_clf.predict_proba(X_test_proceeded)[:,1])
precision_lr,recall_lr, threshold_lr=precision_recall_curve(y_true=y_test,
                                                            probas_pred=model_tree_clf1.predict_proba(X_test_proceeded)[:,1])
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(precision_nb, recall_nb,label='decisiontree')
ax.plot(precision_lr, recall_lr,label='randomforest')
ax.set_xlabel=('Precision')
ax.set_ylabel=('recall')
ax.set_title('precision_recall_curve')
ax.legend()
ax.grid();

#adjust precision and recall for random forest
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(threshold_lr,precision_lr[1:],label='Precision')
ax.plot(threshold_lr,recall_lr[1:],label='Recall')
ax.set_xlabel('classification thresholds')
ax.set_ylabel('precision,recall')
ax.set_title('randomforest: precision,recall')
ax.hline(y=0.9,xmin=0,xmax=1,color='red')
ax.legend()
ax.grid();

#classifier with threshold of 0.2
y_pred_proba=model_tree_clf.predict_proba(X_test_proceeded)[:,1]
y_pred_test=(y_pred_proba>=0.2).astype('int')
#confusion matrix
CM=confusion_matrix(y_pred=pred,y_true=y_test)
print('recall:',100*recall_score(y_pred=y_pred_test,y_true=y_test))
print('precision:',100*precision_score(y_pred=y_pred_test,y_true=y_test))
print(CM)

#making individual prediction
def make_ind_prediction(new_data):
    data=new_data.values.reshape(1,-1)
    robust_scaler=RobustScaler()
    data=robust_scaler.transform(data)
    prob=model_tree_clf1.fit(data)[0][1]
    if prob>=0.2:
        return '0'
    else:
        return '1'
    
#pay=default[default['default']==0]
#pay.head()
from collections import OrderedDict
new_customer=OrderedDict([('Sex_ordered',0.217105),('age',55),('BP',111),
                          ('Blood Suger(rbs)',125),
                          ('bmi',25),
                          ('area_ordered',0.24911)])
new_customer=pd.Series(new_customer)
make_ind_prediction(new_customer)
