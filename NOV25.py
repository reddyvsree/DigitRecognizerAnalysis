# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 14:07:30 2018

@author: CHANDU
"""
#--------------------------------------------importinng session 
import dill
dill.load_session('digit.pkl')

import os
import numpy as np
import pandas as pd

#working directory
os.chdir('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\DIGIT RECOGNIZER')
digit_train= pd.read_csv('train.csv')
digit_test=pd.read_csv('test.csv')

# column names
#summary
summary= digit_train.describe()
digit_train.info()

def missing(x):
    return sum(x.isnull())

digit_train.apply(missing,axis=0)

#data 
X= digit_train.drop('label',axis=1)
Y= digit_train['label']

#--------------------------------splitting data into---------
from sklearn.model_selection import train_test_split
#-----------------------------test zize= 20%--random state is seed value-----
#-----------------------also its random splitting of 80% train  20% test------
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size= 0.2,random_state= 434)

#-----------------------svm for x_train y_train-------------------
from sklearn import svm
sv= svm.SVC(kernel='rbf')

model_sv= sv.fit(x_train,y_train)
predicted_sv= model_sv.predict(x_train)
confusion_x_train= confusion_matrix(y_train,predicted_sv)
accurcay_x_train_score= accuracy_score(y_train,predicted_sv)
print(accurcay_x_train_score)

#---------------------svm for x_test y_test------------

predicted_sv_split_test= model_sv.predict(x_test)
confusion_matrix_split_test= confusion_matrix(y_test,predicted_sv_split_test)
accucary_split_test=accuracy_score(y_test,predicted_sv_split_test)
print(accucary_split_test)

#-----------------confusion matrix accucary------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


X.Loc['Item_Weight]

#----fit to   x_test,y_test-------for svm-------------------------------------
predicted_sv_test= model_sv.predict(digit_test)
output= 
output.



#-----------------------MLP CLASSIFIER RUN AT HOME-----------------------------
#HIDDEN LAYER 30,30,30 solver= sgd/'adam' epochs
#------from sklearn.metrices confusing_matrix, accuracy_score--------
#-------if rows increase invcrease layers  and if column/data increase increase increase neurons-- 


#store the output as dataframe
# .to_csv



import dill
filename = 'digit.pkl'
dill.dump_session(filename)
