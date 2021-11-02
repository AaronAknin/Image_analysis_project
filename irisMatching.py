
#irisMatching
#Fisher linear discriminant: reduce the dimensionality of the feature vector
import numpy as np
import pandas as pd
import random, sys, itertools
from scipy import spatial

#V = featureextraction.filter_extraction(imgName, 1)

#implement Fisher linear discriminant for dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid


def matching(X_train, y_train):  
    sklearn_lda = LDA()
    x_lda = sklearn_lda.fit_transform(X_train, y_train)
    model = NearestCentroid()
    model.fit(X_lda, y_train.values.ravel())
    return model


'''
#Q:In the process of Fisher LD, do we first calculate the mean of 7 classes,
#and then check the ditance of each data point and the mean of 7 classes,
#pick the smallest distance and assign the corresponding label(class number) to the data point.
from sklearn.model_selection import train_test_split
#import pandas as pd

#getting the means of each observation
def feature_m(f):
    feature_m = []
    i = 0
    while i < len(f):
        feature_m.append(f[i])
        i = i+2
    return feature_m

#according to the paper, we classify X into 4 clases:clear, occlued, motion blurred, defocused image
#Q:Where to pass the number 4?
X = f
y = numpy.ones(f.shape[0])

#one third of image are used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2/3, shuffle = True, random_state = 0)

#train the classifier
model = NearestCentroid()
model.fit(X_train, y_train.values.ravel())

# Printing Accuracy on Training and Test sets
print(f"Training Set Score : {model.score(X_train, y_train) * 100} %")
print(f"Test Set Score : {model.score(X_test, y_test) * 100} %")


#Fisher linear discriminant searches for projected vectors that best discriminate 
#different classes in terms of maximizing the ratio of between-class to withinclass scatter
class Fisher:
    def _init_(self, data, num_dim):
        self.data = data
        self.num_dim = num_dim
        self.group_feature_byClasses()
        self.calculate_means()
        self.calculate_bc_wc_covs()
        self.calculate_eigenvalues()
        
#Q:Seems nearest center classifier described in the paper is to find the within class distance,
#do we need to find the maximum between class distance?
# 1.separate feature into 108 classes
#2.f and fi are from unknown sample and the ith class? How to choose this "unknown sample"?
#How to choose which ith class? How to make sure the size of f and fi are equivalent?
#3.calculate the L1, L2 distance and cosine similarity measure
#4.find m, which is the mth class that minimizes the distance from f to fi

#group feature into 108 classes by dictionary, each class include one person's eyes(7 images),
#which are 14 numbers from V
def group_feature_byClasses(f):
    grouped_f = {}
    i, j = 0
    while j < len(f):
        for k in range(14):
            grouped_f[i].append(f[j+k])
            j++
        i++
    return grouped_f  

        
def feature_var(f):
    feature_var = []
    i = 1
    while i < len(f):
        feature_mean.append(f[i])
        i = i+2
    return feature_var
        
#calculate L1 distance
def L1_distance(f, fi):
    return sum(abs(a - b) for a, b in zip(f, fi))
    #return sum(abs(a - b))

#calculate L2 distance
def L2_distance(f, fi):
    return sum(sqrt(a - b) for a, b in zip(f, fi))

#calculate cosine distance
def cosine_distance(f, fi):
    return spatial.distance.cosine(f, fi)

def find_m(f):
    feature_mean = feature_mean(f)
    feature_var = feature_var(f)
    grouped_f = group_feature_byClasses(f)
    
    overall_mean = np.mean(feature_mean)#make overall_mean a list of 7 elements
    
    #find the m that mth class has the shortest distance from overall mean
    m1, m2, m3 = {}
    for k in grouped_f.keys():
        m1[k] = L1_distance(overall_mean, grouped_f[k])
        m2[k] = L1_distance(overall_mean, grouped_f[k])
        m3[k] = cosine_distance(overall_mean, grouped_f[k])
    #find out the minimum value in m1, m2, m3 and its coresponding key
'''
    
    