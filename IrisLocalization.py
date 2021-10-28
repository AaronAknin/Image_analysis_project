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
    square = im_gray[max(0,centroid[1]-100):min(centroid[1]+100,width-1),max(0,centroid[0]-100):min(centroid[0]+100,length-1)]
    return square

im_gray = cv2.imread('CASIA Iris Image Database (version 1.0)/001/1/001_1_1.bmp', cv2.IMREAD_GRAYSCALE)
C_gray = centroid(im_gray)
square_120 = resize(im_gray,C_gray)
(thresh, blackAndWhiteImage) = cv2.threshold(square_120, 50, 255, cv2.THRESH_BINARY)
C_binary = centroid(blackAndWhiteImage)
plt.scatter(C_binary[0],C_binary[1])
plt.imshow(square_120)

blurred = cv2.GaussianBlur(square_120, (5, 5), 0)
cv2.imshow('a',blurred)
cv2.waitKey(0)

circles = cv2.HoughCircles(wide,cv2.HOUGH_GRADIENT,1,50,param1=50,param2=10,minRadius=0,maxRadius=100)
circles = np.round(circles[0, :]).astype("int")
for (x, y, r) in circles:
    		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(square_120, (x, y), r, (0, 255, 0), 2)

cv2.imshow('detected circles',square_120)
cv2.waitKey(0)