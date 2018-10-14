#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:48:06 2018

@author: srinivas
"""

import numpy as np # moathematical operations
import matplotlib.pyplot as plt #plot charts
import pandas as pd # import and manage data sets

#import datasets
datasets=pd.read_csv('diabetes.csv')# dataset is a variable that stores data
X=datasets.iloc[: , 0:8].values#X contains elements from column 0 to 7
y=datasets.iloc[: , 8].values  # y contain elements of column 8

#dividing the date
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#implementing ANN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
classifier=Sequential()
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))
classifier.add(Dropout(p=0.1))
classifier.add(Dense(output_dim=2,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=50,batch_size=10)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
