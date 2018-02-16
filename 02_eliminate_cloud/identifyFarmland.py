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
# $ python submit.py
# =============================================================================


import cv2
from os.path import join
import os
import sys

import loadData


def func_loadFarmlandMask(tifPathList):
    panel = cv2.imread(tifPathList[0])

    for idx in range(1,len(tifPathList)):
        # 1. Load image.
        img  = cv2.imread(tifPathList[idx])
        # 2. Get average image.
        panel = cv2.addWeighted(panel, 0.4, img, 0.6, 0)

    grayPanel       =cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    ret, binaryPanel=cv2.threshold(grayPanel,1,255,cv2.THRESH_BINARY)

    #cv2.imshow("Ave image", panel)
    #cv2.imshow("Binary image", binaryPanel)
    #cv2.waitKey(0)

    return binaryPanel



def model_get_farmLand_roi():

    ################################################################################
    # Load all files then regex, and check the availiabity
    ################################################################################
    curPath = os.getcwd()
    dirPath = join(curPath, "data")


    tifPathList = loadData.func_loadData(dirPath, ".tif")
    if (len(tifPathList) == 0):
        sys.exit()
    #print("Files you want:")
    #print(tifPathList, "\n")


    if False == loadData.func_isAvailiableData(tifPathList):
        sys.exit()


    ################################################################################
    # Get mask and find farmland.
    ################################################################################

    # 1. Gray image, binary mask.
    farmlandMask = func_loadFarmlandMask(tifPathList)

    # 2. Read src image.(png)
    pngPathList = loadData.func_loadData(dirPath, ".png")
    if (len(pngPathList) == 0):
        sys.exit()
    #print("Files you want:")
    #print(pngPathList, "\n")

    srcImage = cv2.imread(pngPathList[0])

    # 3. Blur
    # -------------------------------------------------------
    # img_medianBlur=cv2.medianBlur(img01,5)
    # img_Blur=cv2.blur(img01,(5,5))
    # img_GaussianBlur=cv2.GaussianBlur(img01,(7,7),0)
    # img_bilateralFilter=cv2.bilateralFilter(img01,40,75,75)

    blurImage  = cv2.GaussianBlur(srcImage, (11,11),0)

    # 4. Extract roi.
    farmlandMask_rgb = cv2.cvtColor(farmlandMask, cv2.COLOR_GRAY2BGR)
    farmland_roi= cv2.bitwise_and(blurImage, farmlandMask_rgb)

    #cv2.imshow("Farmland ROI image", farmland_roi)
    #cv2.waitKey(0)

    return farmland_roi,srcImage

