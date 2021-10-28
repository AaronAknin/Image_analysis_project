import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def centroid(image):
    centroid_x = 0
    centroid_y = 0

    width , length = image.shape

    X = [2000000]*length
    Y = [2000000]*width

    for j in range(width):
        Y[j] = sum(image[j,i] for i in range(length))
        if Y[j] == min(Y):
            centroid_y = j
    for i in range(length):
        X[i] = sum(image[j,i] for j in range(width))
        if X[i] == min(X):
            centroid_x = i  
    return [centroid_x,centroid_y]

def resize(image,centroid):
    width , length = image.shape
    square = image[max(0,centroid[1]-110):min(centroid[1]+110,width-1),max(0,centroid[0]-110):min(centroid[0]+110,length-1)]
    return square

def circles(image):
    im_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    C_gray = centroid(im_gray)
    square_200 = resize(im_gray,C_gray)
    square_200 = cv2.medianBlur(square_200,5)
    (thresh, blackAndWhiteImage) = cv2.threshold(square_200, 50, 255, cv2.THRESH_BINARY)
    
    circles_pupil = cv2.HoughCircles(blackAndWhiteImage,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=10,minRadius=20,maxRadius=100)
    circles_pupil = np.round(circles_pupil[0, :]).astype("int")

    circles_iris = cv2.HoughCircles(square_200,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=30,minRadius=75,maxRadius=130)
    circles_iris = np.round(circles_iris[0, :]).astype("int")

    return circles_pupil[0],circles_iris[0]

def crop(image):
    circles_pupil,circles_iris = circles(image)
    im_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    C_gray = centroid(im_gray)
    img = resize(im_gray,C_gray)

    # draw filled circles in white on black background as masks
    mask1 = np.zeros_like(img)
    mask1 = cv2.circle(mask1, (circles_pupil[0],circles_pupil[1]), circles_pupil[2], (255,255,255), -1)
    mask2 = np.zeros_like(img)
    mask2 = cv2.circle(mask2, (circles_iris[0],circles_iris[1]), circles_iris[2], (255,255,255), -1)

    # subtract masks and make into single channel
    mask = cv2.subtract(mask2, mask1)

    # put mask into alpha channel of input
    result = cv2.bitwise_and(img,img,mask = mask)

    return result


image = crop('CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp')
cv2.imshow('masked image',image)
cv2.waitKey(0)