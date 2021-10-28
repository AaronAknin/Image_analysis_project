import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def ImageEnhancement(normalized):
    enhanced=[]
    for res in normalized:
        res = res.astype(np.uint8)
        im=cv2.equalizeHist(res)
        enhanced.append(im)
    return enhanced