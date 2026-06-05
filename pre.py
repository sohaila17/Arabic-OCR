import cv2
import os
import numpy as np
import math
from scipy import ndimage
import imutils
from skimage.feature import greycomatrix, greycoprops
from skimage import data


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

    #calculate the width and hight of the box 
    width= math.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)
    hight= math.sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)

    #if the box is vertical make the angle 90 to turn it horizontal
    if box[0][0] == box[1][0] and hight > width:
        angle = 90
    
    if box[1][1] >= box[3][1]:
        angle = (angle+90)

    #rotate the image then cut the access background
    rotated = imutils.rotate_bound(img, angle)
    final = cut(rotated)
    cv2.imshow("before", rotated)
    key = cv2.waitKey(0)
    
    return final


fname = 'Image_10_Arabic Transparent_28.png'

# read image as greyscale (0)
image = cv2.imread(fname,0)

#rotation
r = rotation(image)

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

#display the image
cv2.imshow("After", med_denoised)  
key = cv2.waitKey(0)

img = med_denoised

#### feature extraction #####
haralick_labels = ["Angular Second Moment",
                   "Contrast",
                   "Correlation",
                   "Sum of Squares: Variance",
                   "Inverse Difference Moment",
                   "Sum Average",
                   "Sum Variance",
                   "Sum Entropy",
                   "Entropy",
                   "Difference Variance",
                   "Difference Entropy",
                   "Information Measure of Correlation 1",
                   "Information Measure of Correlation 2",
                   "Maximal Correlation Coefficient"]


#
img[img == 255] =1

feature = []

glcm = greycomatrix(img, [5], [0], 2, symmetric=True, normed=True)
#print(glcm.shape)
#print (glcm)
p=np.reshape(glcm, (2,2))
print('p:',p)
L = len(p[0])
k=np.arange(L)
k2=k**2
#print('k:',k2)

#mean
mu=0
for i in range(L):
    for j in range(L):
        mu+= i*p[i][j]
#print('mean:',mu)


#px -> sum of rows / py -> sum of columns
px= p.sum(axis=0)
py= p.sum(axis=1)

varx=0
vary=0

mux = np.dot(px, k)
muy = np.dot(py, k)


px_plus_y= np.zeros((2*L)-1)
px_minus_y = np.zeros(L)



for i in range(L):
    for j in range(L):
        px_plus_y[i+j] += p[i][j]
        px_minus_y[abs(i-j)] += p[i][j]
        
'''
print('px:',px)
print('py:',py)
vx = np.dot(px, k2) - mux**2
vy = np.dot(py, k2) - muy**2
pxAddy = np.zeros((2*L)-1)
pxSuby = np.zeros(L)

for k in range (len(pxAddy)):
    for i in range(L):
        for j in range(L):
            if i+j==k:
                pxAddy[k] += p[i][j]
for k in range (len(pxSuby)):
    for i in range(L):
        for j in range(L):
            if abs(i-j)==k:
                pxSuby[k] += p[i][j]


print('pxAddy:',pxAddy)
print('pxSuby:',pxSuby)
'''

'''
print('vx:',mux)
print('vy:',muy)

print('mux:',mux)
print('muy:',muy)
'''
for i in range(L):
    varx += ((i-mux)**2)*px[i]
    vary += ((i-muy)**2)*py[i]
    
print('varx:',varx)
print('vary:',vary)


        

#ASM angular second moment
ASM = np.sum(p**2)
feature.append(ASM)

#contrast
contrast = greycoprops(glcm, 'contrast')[0][0]
co=0
for i in range(L):
    for j in range(L):
        co += (i-j)**2*p[i][j]
feature.append(contrast)

#correlation
correlation = greycoprops(glcm, 'correlation')[0][0]
feature.append(correlation)

'''
corr=0
for i in range(1,L):
    for j in range(1,L):
        corr += ((i*j)*p[i][j]-(mux*muy))/(np.sqrt(varx)*np.sqrt(vary))
print ('correlation1:',corr)
'''

#sum of squares: variance
variance = 0
for i in range(L):
    for j in range(L):
        variance += (i-mu)**2*p[i][j]
feature.append(variance)


#inverse difference moment
inv = 0
for i in range(L):
    for j in range(L):
        inv += (1/(1+(i+j)**2))*p[i][j]
feature.append(inv)


#sum average
avg = 0
for i in range(len(px_plus_y)):
    avg += i*px_plus_y[i]
feature.append(avg)

#sum variance
muxAddy = np.dot(px_plus_y,np.arange(len(px_plus_y)))
Svar =0
for k in range(len(px_plus_y)):
    Svar += (k-muxAddy)**2*px_plus_y[k]

feature.append(Svar)


#sum entropy
SENT = 0
for i in range(len(px_plus_y)):
    SENT += px_plus_y[i]*np.log(px_plus_y[i])
SENT = SENT*-1
feature.append(SENT)

#entropy
ENT=0
for i in range(L):
    for j in range(L):
        ENT+= p[i][j]*np.log(p[i][j])
ENT= ENT*-1
feature.append(ENT)

#difference variance
muxSuby = np.dot(px_minus_y,np.arange(len(px_minus_y)))
Dvar =0
for k in range(len(px_minus_y)):
    Dvar += (k-muxSuby)**2*px_minus_y[k]

feature.append(Dvar)

#difference entropy
DENT = 0
for i in range(L):
    DENT += px_minus_y[i]*np.log(px_minus_y[i])
DENT = DENT*-1
feature.append(DENT)


#information measure of correlation-IHXY
HXY, HXY1, HXY2, HX, HY = 0, 0, 0, 0, 0
for i in range(L):
    HX += px[i]*np.log(px[i])
    HY += py[i]*np.log(py[i])
    for j in range(L):
        HXY += p[i][j]*np.log(p[i][j])
        HXY1 += p[i][j]*np.log(px[i]*py[j])
        HXY2 += px[i]*py[j]*np.log(px[i]*py[j])
HXY = HXY*-1
HXY1 = HXY1*-1
HXY2 = HXY2*-1
HX = HX*-1
HY = HY*-1

IMC1 = (HXY-HXY1)/(max(HX,HY))
feature.append(IMC1)

#information measure of correlation-II
IMC2 = (1 - math.exp(-2*(HXY2-HXY)))**0.5
feature.append(IMC2)

#the maximal correlation coefficient
Q =[[0,0],[0,0]]
for i in range(L):
    for j in range(L):
        for k in range(L):
            Q[i][j] += (p[i][k]*p[j][k])/(px[i]*py[k])
            
flat= np.asarray(Q).flatten()
flat.sort()
x=flat[-2]
MCC = x**0.5
feature.append(MCC)

for i in range(len(feature)):
    print(haralick_labels[i],' ',feature[i])

