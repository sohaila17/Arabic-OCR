import cv2
import numpy as np
import math
from scipy import ndimage
import imutils




def cut(img):
    tol = 50
    mask = img>tol
    img = img[np.ix_(mask.any(1),mask.any(0))]
    return img


def rotation(img):
    #get the angle (0,-90] of rotation and the points of the rectanglr
    #that contains the text 
    coords = np.column_stack(np.where(img > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    if box[1][1] >= box[3][1]:
        angle = (angle+90)

    #rotate the image then cut the access background
    rotated = imutils.rotate_bound(img, angle)
    final = cut(rotated)
    
    
    return final

def pre_processing(img):
    #rotation
    r = rotation(img)

    #resize
    dim = (80, 60)
    resized = cv2.resize(r, dim, interpolation = cv2.INTER_AREA)

    # convert image to binary using adaptive thresh mean
    th1 = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    #morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # remove noise using median filter (image, size)
    med_denoised = ndimage.median_filter(opening, 3)
    
    return med_denoised
