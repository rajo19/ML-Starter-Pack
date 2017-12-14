# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:54:27 2017

@author: rajor
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#loading the data
iris=load_iris()
X=iris.data
y=iris.target
z=iris.target_names
from sklearn.neighbors import KNeighborsClassifier
k=KNeighborsClassifier(n_neighbors=1)
k.fit(X,y)
#predicting a sample 
print("Predict for a sample:")
X_new= [0,0,0,0]
for i in range(4):
    X_new[i]=input()
X_new=np.array(X_new)
X_new=X_new.reshape(1, -1)
a=k.predict(X_new)
if (a[0]==0):
    print(z[0])
elif (a[0]==1):
    print(z[1])
elif (a[0]==2):
    print(z[2])
#plotting the iris dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')






