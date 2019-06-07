#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 02:59:55 2019

@author: ajay
"""

# -*- coding: utf-8 -*-
"""
PIC 16 Spring 2019
Startup code for homework 8
"""

from scipy.misc import imread # using scipy's imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn import svm
import pickle


#pass the boundaries into SVM estimator(uses boundaries to generate hyperplane, a line of best fit)
def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)
def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped
def resize_image(imgs,dim):
    img_re=[]
    for img in imgs:
        img = resize(img,dim)
        img_re.append(img)
    return img_re


def load_data(datasets):
    data=None
    for parseFile,target_val in datasets:
        big_img = imread(parseFile,flatten=True)
        imgs = separate(big_img) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)
        resized=resize_image(imgs,(10,10))
        for img in resized:
            plt.imshow(img, cmap='gray')
            plt.show()
        resized=np.asarray(resized)
        new_data=resized.reshape(len(imgs),-1)
        new_target=np.ones(new_data.shape[0])+target_val
        if data is None:
             data=new_data
             target=new_target
        else:
             data=np.concatenate((data,new_data))
             target=np.concatenate((target,new_target))
        
    return data,target
def partition(data,target,p):
    #p=percentage of data to train on
    rand_num=np.random.rand(data.shape[0])
    train=rand_num<p
    test=np.logical_not(train)
    train_data,train_target=data[train,:],target[train]
    test_data,test_target=data[test,:],target[test]
    return train_data, train_target, test_data, test_target

if __name__ == "__main__":
    datasets=[('/Users/ajay/Desktop/a.png',0),('/Users/ajay/Desktop/b.png',1),('/Users/ajay/Desktop/c.png',2)]
    new_data,new_target=load_data(datasets)
    p=0.8
    train_data,train_target, test_data, test_target= partition(new_data, new_target,p)
    #create SVC estimator, pass in train data and target element to train, pass in a new test_data to predict
    clf=svm.SVC(gamma=0.1)
    clf.fit(train_data,train_target)
    predicted=clf.predict(test_data)
    
    #persisted learning
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    predicted=clf2.predict(test_data)
    print "Predicted: ", predicted


'''
#step 1: take pictures of handwriting of letters
upload three different images, file name 
map all a arrays to target array of 0's, 1's for bs, 2's for c's 
concatenate target array for all 
when u predict, u reference this target array?




#write a b c in multiple ways(see example) as training data
#step 2: create data array using imread
#step 3: create target array using imread to test data
#step 4: create SVC estimator- clf = svm.SVC(gamma=0.001, C=100.)
#step 5: "fit" data set to target set using SVC-clf.fit(digits.data[:-1], digits.target[:-1]) 
 # data[:-1]- everything but last element, we save last element for "test" data
#step 6: "predict" an uknown data set using SVC estimator- clf.predict(digits.data[-1:])
 #this case- using last element in data set to use as test data, pass into SVC to predict best guess
 '''