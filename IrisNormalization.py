import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def IrisNormalization(image, height, width, circles_pupil, circles_iris):     
    theta_list = np.arange(0, 2 * np.pi, 2 * np.pi / width)

    empty = np.zeros((height,width, 3), np.uint8)

    circle_x = circles_pupil[1]
    circle_y = circles_pupil[0]

    r_pupil = circles_pupil[2]
    r_iris = circles_iris[2]
    color = [0,0,0]
    for i in range(width):
        for j in range(height):
            theta = theta_list[i]
            r = j / height

            Xp = circle_x + r_pupil * np.cos(theta)
            Yp = circle_y + r_pupil * np.sin(theta)
            Xi = circle_x + r_iris * np.cos(theta)
            Yi = circle_y + r_iris * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            X = Xp + ( Xi - Xp )*r
            Y = Yp + ( Yi - Yp )*r

            shapes = image.shape
            if X < shapes[0] and Y < shapes[1]:
                color = image[int(X)][int(Y)]  # color of the pixel
            empty[j][i] = color
    return empty
