import cv2
import numpy as np

def ImageEnhancement(image):
    output = []
    for line in image:
        line = line.astype(np.uint8)
        im = cv2.equalizeHist(line)
        output.append(im)
    return np.array([output[i] for i in range(len(output))])