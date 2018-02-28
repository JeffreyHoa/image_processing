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
#from matplotlib import pyplot as plt

import myTools
import dnn_segment
import gmm_segment
import identifyFarmland
import identifyCrop
import identifyShadow
import identifyCloud

debugger = True

# Get mask images of crop, shadow and cloud.
farmland_roi, src_img     = identifyFarmland.model_get_farmLand_roi()
crop_mask                 = identifyCrop.model_get_crop_mask(farmland_roi)
shadow_mask, shadow_exist = identifyShadow.model_get_shadow_mask(farmland_roi)
cloud_mask                = identifyCloud.model_get_cloud_mask()

# Fusion of negative samples: shadow and cloud.
negative_sample = cv2.addWeighted(shadow_mask, 0.5, cloud_mask, 0.5, 0)
ret, negative_sample = cv2.threshold(negative_sample, 1, 255, cv2.THRESH_BINARY)
neg_pos_sample = cv2.addWeighted(negative_sample, 0.7, crop_mask, 0.3, 0)

neg_pos_sample_rgb = cv2.cvtColor(neg_pos_sample, cv2.COLOR_GRAY2BGR)
neg_pos_sample_overSrc = cv2.addWeighted(src_img, 0.8, neg_pos_sample_rgb, 0.2, 0)

# Debug show.
#cv2.imshow("crop_mask",   crop_mask)
#cv2.imshow("shadow_mask", shadow_mask)
#cv2.imshow("cloud_mask",  cloud_mask)
#cv2.imshow("neg and pos fusion", neg_pos_sample)
#cv2.imshow("neg and pos fusion over src", neg_pos_sample_overSrc)
#cv2.waitKey(0)


####################
# option.
# dnn or gmm.
####################
def dnn_binSeg(dnnSeg_cnt=1):

    g_step_size = 10
    dnn_crop_mask = crop_mask.copy()

    for i in range(0, dnnSeg_cnt):

        ###############################################################################
        # Build train data.
        ###############################################################################
        X_ret1, Y_ret1 = myTools.build_trainSample(dnn_crop_mask,   g_step_size, True)
        X_ret2, Y_ret2 = myTools.build_trainSample(shadow_mask, g_step_size, False)
        X_ret3, Y_ret3 = myTools.build_trainSample(cloud_mask,  g_step_size, False)

        X_list =  X_ret1 + X_ret2 + X_ret3
        Y_list =  Y_ret1 + Y_ret2 + Y_ret3

        # Train Data: translate list to array.
        X = np.array(X_list)
        Y = np.array(Y_list)

        #print("Num of sample", len(X_list))
        #print("X:", X)
        #print("Y:", Y)


        ###############################################################################
        # Train and segment.
        ###############################################################################
        segmentRstName = "segmentRst.png"
        rows = farmland_roi.shape[0]
        cols = farmland_roi.shape[1]

        dnn_segment.func_dnn_segment(X,Y, segmentRstName, rows, cols)


        ###############################################################################
        # Load segmentRst and get boundary mask.
        ###############################################################################
        boundary_mask = myTools.func_extractBoundary(segmentRstName)


        ###############################################################################
        # Trim crop area edge.
        ###############################################################################
        dnn_crop_mask = myTools.func_cleanMask(dnn_crop_mask, boundary_mask)
        title_debug = "Crop mask (after opt): " + str(i)
        cv2.imshow(title_debug, dnn_crop_mask)
        cv2.waitKey(0)

        # Only retain the largest part.
        _, contours, hierarchy = cv2.findContours(dnn_crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bigestContour = myTools.contourSizeFiltering(contours)
        ret, dnn_crop_mask = cv2.threshold(dnn_crop_mask, 0, 0, cv2.THRESH_BINARY)
        cv2.fillPoly(dnn_crop_mask, pts =bigestContour, color=255)

        # Now, after dnn optimization, we get a new crop mask.
        return dnn_crop_mask


crop_mask = dnn_binSeg()

crop_mask_rgb = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
result_img    = cv2.addWeighted(src_img, 0.8, crop_mask_rgb, 0.2, 0)

cv2.imwrite("CropMask.png", crop_mask)
cv2.imshow("Result",  result_img)
cv2.waitKey(0)



###############################################################################
# GMM
###############################################################################

def findWrapContour(mask):

    contour_img, trainData = gmm_segment.func_getContours(mask)

    singleChannelImg, colorList = gmm_segment.func_statContour(contour_img)

    wrapContour = gmm_segment.func_findWrapContour(singleChannelImg, colorList, trainData)
    if not wrapContour == []:
        cv2.drawContours(result_img, wrapContour, -1, 50, 5)
        cv2.imshow("Result",  result_img)
        cv2.waitKey(0)

    return wrapContour


if shadow_exist == True:
    shadow_wrapContour = findWrapContour(shadow_mask)
    cv2.drawContours(crop_mask, shadow_wrapContour, -1, 0, 5)

cloud_wrapContour  = findWrapContour(cloud_mask)
cv2.drawContours(crop_mask, cloud_wrapContour, -1, 0, 5)


# Only retain the largest part.
_, contours, hierarchy = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
bigestContour = myTools.contourSizeFiltering(contours)
ret, crop_mask = cv2.threshold(crop_mask, 0, 0, cv2.THRESH_BINARY)
cv2.fillPoly(crop_mask, pts =bigestContour, color=255)


cv2.imwrite("CropMask.png", crop_mask)
cv2.imshow("Result of Crop",  crop_mask)
cv2.waitKey(0)








