import os

#for loop read in data 
dir = ("/Users/wuyin/Grad/21Fall/5293-001_ImageAnalysis/GroupProj")
x_train, X_test = [], []
for file in dir:
    img_train = os.path.join(dir, file, '1/.img')
    img_test = os.path.join(dir, file, '2/.img')
    img_localized_train = irisLocalization.centroid(img_train)
    img_normalized_train = irisNormalization.daugman_normalizaiton(img_localized_train)
    X_train.append(imgeEnhancement(img_normalized_train))
    img_localized_test = irisLocalization.centroid(img_test)
    img_normalized_test = irisNormalization.daugman_normalizaiton(img_localized_test)
    X_test.append(imgeEnhancement(img_normalized_test))
#training set includes the first session, and testing set includes the second session

#go through localization, normalization, enhancement and featureExtraction

