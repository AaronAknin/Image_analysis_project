import cv2
import numpy as np
import glob
import math
import scipy
from scipy.spatial import distance
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation
import warnings
warnings.filterwarnings("ignore")

def function(folder_path):
    '''TRAINING'''

    #reading the training images from the CASIA dataset
    feature_vector_train = []
    class_train = []
    for i in range(1,109):
        for j in range(1,4):
            path = folder_path+"/"+'{:03}'.format(i)+'/1/'+'{:03}'.format(i)+'_1_'+str(j)+'.bmp'
            print(path)
            a = IrisLocalization(path)
            b = IrisNormalization(a[0], 64, 512, a[1], a[2])
            c = ImageEnhancement(b)
            d = FeatureExtraction(c)
            feature_vector_train.append(d)
            class_train.append(i)
    print("Training data processed.")


    '''TESTING'''

    feature_vector_test = []
    class_test = []
    for i in range(1,109):
        for j in range(1,5):
            path = folder_path+"/"+'{:03}'.format(i)+'/2/'+'{:03}'.format(i)+'_2_'+str(j)+'.bmp'
            print(path)
            a = IrisLocalization(path)
            b = IrisNormalization(a[0], 64, 512, a[1], a[2])
            c = ImageEnhancement(b)
            d = FeatureExtraction(c)
            feature_vector_test.append(d)
            class_test.append(i)
    print("Testing data processed.")

    components = [15,30,60,90,120,160,200]
    L = []
    for component in components:
        predict = IrisMatching(feature_vector_train,feature_vector_test,component)
        performance = PerformanceEvaluation(predict,class_test)
        L.append([component,performance])

    plt.plot(components,[L[i][1][0] for i in range(len(L))])
    plt.show()

if __name__ == "__main__":
    folder_path = "CASIA Iris Image Database (version 1.0)"
    function(folder_path)

