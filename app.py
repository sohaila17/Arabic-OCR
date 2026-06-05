import cv2
import os
import numpy as np
import preprocessing as pre
import glcm
import knn
import genetic as ga
from collections import Counter
import fknn
from sklearn.model_selection import train_test_split, cross_val_score



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # Using 0 to read image in grayscale mode
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            img_after = pre.pre_processing(img)
            images.append(img_after)
            
    final = np.asarray(images)
    
    return final

im = load_images_from_folder('image')

#load the labels from the labels file
filename = "labels.dat"
labels = [i.strip() for i in open("labels.dat",encoding='utf-8').readlines()]

'''
for i in range(len(labels)):
    cv2.imshow("after", im[i])
    key = cv2.waitKey(0)
    print('The word is: ',labels[i])
'''
haralick_labels = []
features = []

#get the glcm features to every image
for i in range(len(im)):
    haralick_labels, feature = glcm.glcm(im[i,:,:])
    features.append(feature)
   
features = np.asarray(features)

print('start')

#feature selection
feature = ga.feature_selection(features,labels)
c = Counter(feature)
print(feature)
print(c)

#get the index of the selected features
indx = [i for i, x in enumerate(feature) if x]
print(indx)

new_features = features[:,indx]
print(new_features)
    
for i in range(len(haralick_labels)):
    if feature[i]==True:
        print (haralick_labels[i])


xTrain, xTest, yTrain, yTest = train_test_split(new_features,labels)

custModel = fknn.FuzzyKNN()

scores = cross_val_score(cv=4,estimator=cusmodel, X=xTest,Y=yTest)
    
print(scores)
