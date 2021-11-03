import numpy as np
import pylab
import cv2
import math
import matplotlib.pyplot as plt

def filter_extraction(imagename, i):
    # imagename shows the image_ROI
    # i shows the channel of the spatial filters, i=1,2
    # For the frequency f, you can assign f as 1/deltaY. 
    # You can also set f to 1/deltaX to compare the results.
    def filter_Ma(i):
        # delta_x = 3 and delta_y = 1.5 in paper Ma et al
        delta_x = [3, 4.5]
        delta_y = [1.5, 1.5]
        f = 1/delta_x[i-1]
        # Bounding box
        nstds = 3  # Number of standard deviation sigma
        ## Set x,y using three sigma rule
        theta = 0  
        # we set theta as zero, because the horizontal direction contains the higer information density
        xmax = max(abs(nstds * delta_x[i-1] * np.cos(theta)), abs(nstds * delta_y[i-1] * np.sin(theta)))
        xmax = np.ceil(max(1, xmax))
        ymax = max(abs(nstds * delta_x[i-1] * np.sin(theta)), abs(nstds * delta_y[i-1] * np.cos(theta)))
        ymax = np.ceil(max(1, ymax))
        xmin = -xmax
        ymin = -ymax
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

        # Modulating function M: a circularly symmetric sinusoidal function
        M = np.cos(2 * np.pi * f * np.sqrt(x ** 2 + y ** 2))
        fm = (1/(2 * np.pi * delta_x[i-1] * delta_y[i-1])) * np.exp(-.5 * ((x / delta_x[i-1]) ** 2 + (y / delta_y[i-1]) ** 2)) * M
        return fm
    ## Convolution Part ##
    img = plt.imread(imagename)  # not sure the output of the format
    #I=cv2.imread(imagename) # Color image
    I=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(img)                                   
    #  pylab.show()
    # convolution and use f(0,1/delta_x))
    def F(i):
        return cv2.filter2D(I,-1,filter_Ma(i))
        #F[i] = cv2.filter2D(I,-1,filter_Ma_1(0)) # take theta = 0
        #F_2 = cv2.filter2D(I,-1,filter_Ma_2(0)) # take theta = 0
        
    # picture after convolution
    '''
    plt.imshow(F_1)
    pylab.show()
    plt.imshow(F_2)
    plt.imsave("F_1.png",F_1)
    plt.imsave("F_2.png",F_2)
    pylab.show()
    '''
    ## 2D filter encode
    ### Here we need to get the size of image I (Iheight, Iwidth)
    v = []  ## feature vector
    # encode filtered image
    # def filter_encode(filter_Ma(i)): 
    # imagename can be chosen between F1 and F2
    # filter can be chosen in filter_Ma_1 and filter_Ma_2
    # filter_num = 2
    # print(I.shape)
    I_height = F(i).shape[0]
    I_width = F(i).shape[1]
    # filter size: set filter height and width, such as 9*9
    filter_height = 9
    filter_width = 9
    mmax = math.floor(I_height/filter_height)
    nmax = math.floor(I_width/filter_width)
    
    # iteration: for k in range(int(mmax * nmax))
    
    for m in range(int(mmax)):
        for n in range(int(nmax)):
            # do partition from the image and compute inner product
            part_F = F(i)[m*filter_height:(m+1)*filter_height, n*filter_width:(n+1)*filter_width]
            avg = np.mean(part_F)
            sigma = np.mean(part_F - avg)
            v.append(avg)
            v.append(sigma)
            
    #print(v)  # feature vector
    return v

# filter_extraction('/Users/yangwenqing/Desktop/unnamed-1.png',1)