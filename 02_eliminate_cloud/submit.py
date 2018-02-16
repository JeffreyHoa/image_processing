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
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

import myTools
import dnn_segment
import identifyFarmland
import identifyCrop
import identifyShadow
import identifyCloud

debugger = True

# Get mask images of crop, shadow and cloud.
farmland_roi, src_img = identifyFarmland.model_get_farmLand_roi()
crop_mask    = identifyCrop.model_get_crop_mask(farmland_roi)
shadow_mask  = identifyShadow.model_get_shadow_mask(farmland_roi)
cloud_mask   = identifyCloud.model_get_cloud_mask()

# Fusion of negative samples: shadow and cloud.
negative_sample = cv2.addWeighted(shadow_mask, 0.5, cloud_mask, 0.5, 0)
ret, negative_sample = cv2.threshold(negative_sample, 1, 255, cv2.THRESH_BINARY)
neg_pos_sample = cv2.addWeighted(negative_sample, 0.7, crop_mask, 0.3, 0)

# Debug show.
#cv2.imshow("crop_mask",   crop_mask)
#cv2.imshow("shadow_mask", shadow_mask)
#cv2.imshow("cloud_mask",  cloud_mask)
cv2.imshow("neg and pos fusion", neg_pos_sample)
#cv2.waitKey(0)



###############################################################################
# Build train data.
###############################################################################

g_step_size = 10

X_ret1, Y_ret1 = myTools.build_trainSample(crop_mask,   g_step_size, True)
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
boundary_mask     = myTools.func_extractBoundary(segmentRstName)
boundary_mask_rgb = cv2.cvtColor(boundary_mask, cv2.COLOR_GRAY2BGR)
result_img        = cv2.addWeighted(src_img, 0.9, boundary_mask_rgb, 0.1, 0)

cv2.imshow("Result", result_img)
cv2.waitKey(0)



