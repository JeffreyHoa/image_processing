# Copyright 2018 Jeffrey Hoa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# How to run it?
# $ python main.py
# =============================================================================

import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

import os
import sys
import re



def func_cleanMask(grayImg, maskImg):

    for row_idx in range(len(maskImg)):
        for col_idx in range(len(maskImg[row_idx])):
            if maskImg[row_idx][col_idx] == 255:
                grayImg[row_idx][col_idx] = 0

    return grayImg


###############################################################################
# Load segmentRst and get boundary mask.
###############################################################################
def func_extractBoundary(img):

    plotImg = cv2.imread(img)


    # Identify boundary only based on red channel (white color)
    hist_rgb_list = func_colorHist(plotImg)

    hist_redChannel = hist_rgb_list[2]
    hist_redChannel_flatten = hist_redChannel.flatten()

    rev_red_value_list = list(hist_redChannel_flatten)
    rev_red_value_list.reverse()
    boundaryColor_pixelcnt = next(x for x in rev_red_value_list if x > 0)
    boundaryColor_rev_idx = rev_red_value_list.index(boundaryColor_pixelcnt)

    boundary_redChannel_value = len(rev_red_value_list) - 1 - boundaryColor_rev_idx
    #print("Debug:", boundary_redChannel_value)


    _, _, boundary_redChannel_image = cv2.split(plotImg)
    ret, boundary_mask = cv2.threshold(boundary_redChannel_image, boundary_redChannel_value-1, 255, cv2.THRESH_BINARY)

    # Jeff: plot has different axis with image format, so transfer this here.
    # http://blog.csdn.net/sunjinshengli/article/details/78110946http://blog.csdn.net/sunjinshengli/article/details/78110946
    #cv2.imshow("Boundary mask1", boundary_mask)
    boundary_mask = boundary_mask[::-1]
    boundary_mask = boundary_mask.T

    #cv2.imshow("Boundary mask2", boundary_mask)
    #cv2.waitKey(0)

    return boundary_mask




################################################################################
# Find the main color.
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
################################################################################

def func_grayHist(img ):
    hist = plt.hist(img.ravel(), bins=256, range=[0, 256]);
    plt.show()
    return hist


def func_colorHist(img, show=False):
    color = ('b', 'g', 'r')
    hist_rgb_list = []
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist_rgb_list.append(hist)
        #plt.plot(hist, color=col)

    if show==True:
        plt.xlim([0, 256])
        plt.show()

    return hist_rgb_list



def func_applyKnn(img, K):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Jeff: Normally, 5 types of colors: shadow, cloud, shattered cloud, background, foreground.
    ret,label,center=cv2.kmeans(Z,K,None,criteria,5,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res    = center[label.flatten()]
    res2   = res.reshape((img.shape))

    #cv2.imshow('knn result', res2)
    return res2


def contourSizeFiltering(contours):
    """
    this function filters out the smaller retroreflector (as well as any noise) by size
    """

    if len(contours) == 0:
        print ("sizeFiltering: Error, no contours found")
        return 0

    big = contours[0]


    for c in contours:
        if type(c) and type(big) == np.ndarray:
            if cv2.contourArea(c) > cv2.contourArea(big):
                big = c
        else:
            print(type(c) and type(big))
            return 0
    #x,y,w,h = cv2.boundingRect(big)


    # Now, get the biggest one.
    bigContour = []
    bigContour.append(big)

    return bigContour



def cal_centroid4contour(contour):
    M=cv2.moments(contour)
    print(M)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    print(cx, cy)

    return (cx, cy)




def build_trainSample(img, step_size, positive=True):
    X = []
    Y = []

    for row_idx in range(0, img.shape[0], step_size):
        for col_idx in range(0, img.shape[1], step_size):

            if img[row_idx][col_idx] > 0:
                X.append([row_idx, col_idx])
                Y.append(positive)
            else:
                pass;
    return X,Y



def func_getSingleColorMask(red_ch, color_value):
    red_roi = red_ch.copy()

    for row_idx in range(red_ch.shape[0]):
        for col_idx in range(red_ch.shape[1]):
            if (red_roi[row_idx][col_idx] == color_value):
                red_roi[row_idx][col_idx] = 255
            else:
                red_roi[row_idx][col_idx] = 0

    return red_roi




