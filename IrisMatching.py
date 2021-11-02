import cv2
import numpy as np
import glob
import math
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def dim_reduction(list_train, list_test, components):
    
    #get the classes of all training feature vectors
    class_train = []
    for i in range(1,109):
        for j in range(0,3):
            class_train.append(i)
    class_train = np.array(class_train)
    
    #fit the LDA model on training data with n components
    lda = LinearDiscriminantAnalysis(n_components=components)
    lda.fit(list_train, class_train)
    
    #transform the traning data
    reduced_train = lda.transform(list_train)
    
    #transform the testing data
    reduced_test = lda.transform(list_test)
    
    #return transformed training and testing data, and the testing classes and predicted values for ROC
    return reduced_train , reduced_test, class_train

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)

def L1_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += np.abs(row1[i] - row2[i])
    return distance

def L2_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return distance

def Cosine_distance(row1,row2):
    return 1 - np.dot(row1,row2)/(euclidean_distance(row1,[0]*len(row1))*euclidean_distance(row2,[0]*len(row2)))


def IrisMatching(list_train, list_test, components):
    reduced_train , reduced_test , class_train = dim_reduction(list_train, list_test, components)
    predict = []
    for img_test in reduced_test:
        L1 = []
        L2 = []
        Cosine = []
        for img_train in reduced_train:
            L1.append(L1_distance(img_test,img_train))
            L2.append(L1_distance(img_test,img_train))
            Cosine.append(L1_distance(img_test,img_train))
        mL1 = class_train[L1.index(min(L1))]
        mL2 = class_train[L2.index(min(L2))]
        mCosine = class_train[Cosine.index(min(Cosine))]
        predict.append([mL1,mL2,mCosine])
    return predict