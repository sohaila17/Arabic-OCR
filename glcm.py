import numpy as np
import math
from skimage.feature import greycomatrix, greycoprops

def ASM(p):
    ASM = np.sum(p**2)
    return ASM

def contrast(p,L):
    contrast=0
    for i in range(L):
        for j in range(L):
            contrast += (i-j)**2*p[i][j]
    return contrast

def correlation(glcm):
    correlation = greycoprops(glcm, 'correlation')[0][0]
    return correlation
def var(p,mu,L):
    variance = 0
    for i in range(L):
        for j in range(L):
            variance += (i-mu)**2*p[i][j]
    return variance

def IDM(p,L):
    inv = 0
    for i in range(L):
        for j in range(L):
            inv += (1/(1+(i+j)**2))*p[i][j]
    return inv

def sum_avg(px_plus_y):
    avg = 0
    for i in range(len(px_plus_y)):
        avg += i*px_plus_y[i]
    return avg

def sum_var(px_plus_y):
    muxAddy = np.dot(px_plus_y,np.arange(len(px_plus_y)))
    Svar =0
    for k in range(len(px_plus_y)):
        Svar += (k-muxAddy)**2*px_plus_y[k]
    return Svar

def sum_ent(px_plus_y):
    SENT = 0
    for i in range(len(px_plus_y)):
        SENT += px_plus_y[i]*np.log(px_plus_y[i])
    SENT = SENT*-1
    return SENT

def entropy(p,L):
    ENT=0
    for i in range(L):
        for j in range(L):
            ENT+= p[i][j]*np.log(p[i][j])
    ENT= ENT*-1
    return ENT

def dif_var(px_minus_y):
    muxSuby = np.dot(px_minus_y,np.arange(len(px_minus_y)))
    Dvar =0
    for k in range(len(px_minus_y)):
        Dvar += (k-muxSuby)**2*px_minus_y[k]
    return Dvar

def dif_ent(px_minus_y):
    DENT = 0
    for i in range(len(px_minus_y)):
        DENT += px_minus_y[i]*np.log(px_minus_y[i])
    DENT = DENT*-1
    return DENT

def IMCI(HXY,HXY1,HHX,HY):
    IMC1 = (HXY-HXY1)/(max(HX,HY))
    return IMC1

def IMCII(HXY2,HXY):
    IMC2 = (1 - math.exp(-2*(HXY2-HXY)))**0.5
    return IMC2

def max_corr(p,px,py,L):
    Q =[[0,0],[0,0]]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                Q[i][j] += (p[i][k]*p[j][k])/(px[i]*py[k])

    # convert 2d array to1d array            
    flat= np.asarray(Q).flatten()
    flat.sort()
    #get the 2nd largest element
    x=flat[-2]
    MCC = x**0.5
    return MCC

def glcm(img):
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

    #convert 255 to 1 to work on 2 grey levels(L)
    img[img == 255] =1

    feature = []

    #get co-occurrence matrix
    glcm = greycomatrix(img, [5], [0], 2, symmetric=True, normed=True)
    #convert the output to a 2d array
    p=np.reshape(glcm, (2,2))
    L = len(p[0])
    k=np.arange(L)
    k2=k**2
    mu=0
    for i in range(L):
        for j in range(L):
            mu+= i*p[i][j]
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
    for i in range(L):
        varx += ((i-mux)**2)*px[i]
        vary += ((i-muy)**2)*py[i]

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
    
    feature.append(ASM(p))
    feature.append(contrast(p,L))
    feature.append(correlation(glcm))
    feature.append(var(p,mu,L))
    feature.append(IDM(p,L))
    feature.append(sum_avg(px_plus_y))
    feature.append(sum_var(px_plus_y))
    feature.append(sum_ent(px_plus_y))
    feature.append(entropy(p,L))
    feature.append(dif_var(px_minus_y))
    feature.append(dif_ent(px_minus_y))
    feature.append(IMCII(HXY2,HXY))
    feature.append(IMCII(HXY2,HXY))
    feature.append(max_corr(p,px,py,L))
    
    return haralick_labels, feature
    
    




