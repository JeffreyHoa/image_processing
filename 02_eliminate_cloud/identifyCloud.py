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


import loadData
import myTools
import identifyFarmland
import identifyCrop
import identifyShadow
import identifyCloud





def model_get_cloud_mask():

    # 1. Construct path.
    curPath = os.getcwd()
    dirPath = join(curPath, "data")

    # 2. Read src image.(png)
    pngPathList = loadData.func_loadData(dirPath, ".png")
    if (len(pngPathList) == 0):
        sys.exit()

    srcImage = cv2.imread(pngPathList[0])

    # 3. Blur
    # -------------------------------------------------------
    # img_medianBlur=cv2.medianBlur(img01,5)
    # img_Blur=cv2.blur(img01,(5,5))
    # img_GaussianBlur=cv2.GaussianBlur(img01,(7,7),0)
    # img_bilateralFilter=cv2.bilateralFilter(img01,40,75,75)

    blurImage  = cv2.GaussianBlur(srcImage, (11,11),0)



	# 4. Find relevant color to cloud. [Here can be improved further]
    src_knnResult = myTools.func_applyKnn(blurImage, 5)
    hist_rgb_list = myTools.func_colorHist(src_knnResult)


    # 4.1 Identify cloud only based on red channel (white color)
    hist_redChannel = hist_rgb_list[2]
    hist_redChannel_flatten = hist_redChannel.flatten()

    rev_red_value_list = list(hist_redChannel_flatten)
    rev_red_value_list.reverse()
    cloudColor_pixelcnt = next(x for x in rev_red_value_list if x > 0)
    cloudColor_rev_idx = rev_red_value_list.index(cloudColor_pixelcnt)

    cloud_redChannel_value = len(rev_red_value_list) - 1 - cloudColor_rev_idx
    #print("Debug: cloud_redChannel_value", cloud_redChannel_value)


    # 4.2 Identify scattered cloud only based on red channel (gray color)
    cloudColor_pixelcnt = next(x for x in rev_red_value_list[cloudColor_rev_idx+1:] if x > 0)
    cloudColor_rev_idx = rev_red_value_list.index(cloudColor_pixelcnt)

    grayCloud_redChannel_value = len(rev_red_value_list) - 1 - cloudColor_rev_idx
    #print("Debug: grayCloud_redChannel_value", grayCloud_redChannel_value)


    #cv2.waitKey(0)


    ################################################################################
    # 5. Get cloud mask.
    ################################################################################

    # 5.1 Get red channel.
    _, _, src_redChannel_knnResult = cv2.split(src_knnResult)


    # 5.2 Get mask for crop.
    cloud_roi = src_redChannel_knnResult.copy()

    for row_idx in range(len(src_redChannel_knnResult)):
        for col_idx in range(len(src_redChannel_knnResult[row_idx])):
            if (src_redChannel_knnResult[row_idx][col_idx] == cloud_redChannel_value) or (src_redChannel_knnResult[row_idx][col_idx] == grayCloud_redChannel_value):
                cloud_roi[row_idx][col_idx] = 255
            else:
                cloud_roi[row_idx][col_idx] = 0


    #cv2.imshow("Cloud ROI.", cloud_roi)
    #cv2.waitKey(0)



    ################################################################################
    # Erose land mask for better performance.
    ###############################################################################n
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(30, 30))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))

    # For eliminating holes.
    eroded=cv2.erode(cloud_roi, kernel2);
    dilated = cv2.dilate(eroded, kernel2)

    # For robust: because this is negative sample.
    dilated=cv2.dilate(dilated, kernel);


    #cv2.imshow("*Crop mask (after erose)", dilated)
    #cv2.waitKey(0)


    # Reference.
    #_, contours, hierarchy = cv2.findContours(cloud_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #bigestContour = myTools.contourSizeFiltering(contours)
    #cv2.drawContours(cloud_roi, bigestContour, -1, 100, 10)

    #cv2.imshow("Cloud mask (with contour)", dilated)
    #cv2.waitKey(0)


    return dilated


