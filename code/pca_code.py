#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:51:10 2017

@author: garima
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris

data = np.loadtxt("../dataset/final_dataset_debut_cleaned.csv", dtype=np.object, delimiter=",")
print data.shape
labels = data[1:,183]
view_count = labels.astype(int)
label =labels.astype(int)

operational_data=data[1:,2:182]


#label defination
for i in range(0 , len(labels)):

    if(label[i]<=20000):
        label[i]=1
    if (label[i]>20000 and label[i]<=60000):
        label[i]=2
    if(label[i]>60000 and label[i]<=300000):
        label[i]=3
    if(label[i]>300000 and label[i]<=3000000):
        label[i]=4
    if(label[i]>3000000):
        label[i]=5



#pca code
pca = decomposition.PCA(n_components=20)
X_r = pca.fit(operational_data).transform(operational_data)


df = pd.DataFrame(X_r)
df["views"]= view_count
df["label"]= label
df.to_csv("final_debut_20.csv")


#visualization
f, axarr = plt.subplots(2, 3)
p=0
q=0
lw = 2
for i  in range (0, len(label)):
    if (label[i]==1):
        axarr[0, 0].scatter(X_r[i, 0], X_r[i, 1], color='red', alpha=.8, lw=lw)
        axarr[1, 2].scatter(X_r[i, 0], X_r[i, 1], color='red', alpha=.8, lw=lw)
    if (label[i]==2):
        axarr[0, 1].scatter(X_r[i, 0], X_r[i, 1], color='blue', alpha=.8, lw=lw)
        axarr[1, 2].scatter(X_r[i, 0], X_r[i, 1], color='blue', alpha=.8, lw=lw)
    if (label[i]==3):
        axarr[0, 2].scatter(X_r[i, 0], X_r[i, 1], color='green', alpha=.8, lw=lw)
        axarr[1, 2].scatter(X_r[i, 0], X_r[i, 1], color='green', alpha=.8, lw=lw)
    if (label[i]==4):
        axarr[1, 0].scatter(X_r[i, 0], X_r[i, 1], color='yellow', alpha=.8, lw=lw)
        axarr[1, 2].scatter(X_r[i, 0], X_r[i, 1], color='yellow', alpha=.8, lw=lw)
    if (label[i]==5):
        axarr[1, 1].scatter(X_r[i, 0], X_r[i, 1], color='black', alpha=.8, lw=lw)
        axarr[1, 2].scatter(X_r[i, 0], X_r[i, 1], color='black', alpha=.8, lw=lw)




plt.show()
