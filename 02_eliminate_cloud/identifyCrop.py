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
import myTools


def model_get_crop_mask(farmland_roi):

    ################################################################################
    # Find the main color value, which is also crop land.
    ################################################################################

    # 1.1 knn, k = 5
    farmland_knnResult = myTools.func_applyKnn(farmland_roi, 5)

    hist_rgb_list = myTools.func_colorHist(farmland_knnResult)

    # 1.2 Identify crop only based on red channel
    hist_redChannel = hist_rgb_list[2]
    #print("Debug: hist_redChannel", hist_redChannel)
    hist_redChannel_flatten = hist_redChannel.flatten()
    red_value_list = list(hist_redChannel_flatten)
    crop_redChannel_value = red_value_list.index(max(red_value_list))
    #print("Debug: crop_redChannel_value", crop_redChannel_value)

    # 1.3 Identify cloud only based on red channel (for test)
    #rev_red_value_list = list(red_value_list)
    #rev_red_value_list.reverse()
    #cloudColor_pixelcnt = next(x for x in rev_red_value_list if x > 0)
    #cloudColor_rev_idx = rev_red_value_list.index(cloudColor_pixelcnt)
    #cloud_redChannel_value = len(rev_red_value_list) - 1 - cloudColor_rev_idx
    #print("Debug: cloud_redChannel_value", cloud_redChannel_value)

    #crop_redChannel_value = cloud_redChannel_value


    ################################################################################
    # Get crop land mask.
    ################################################################################

    # 2.1 Get red channel.
    _, _, farmland_redChannel_knnResult = cv2.split(farmland_knnResult)


    # 2.2 Get mask for crop.
    crop_roi = farmland_redChannel_knnResult.copy()

    for row_idx in range(len(farmland_redChannel_knnResult)):
        for col_idx in range(len(farmland_redChannel_knnResult[row_idx])):
            if not farmland_redChannel_knnResult[row_idx][col_idx] == crop_redChannel_value:
                crop_roi[row_idx][col_idx] = 0
            else:
                crop_roi[row_idx][col_idx] = 255


    #cv2.imshow("Crop ROI.", crop_roi)
    #cv2.waitKey(0)



    ################################################################################
    # Erose land mask for better performance.
    ###############################################################################n

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(30, 30))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

    # For robust
    #eroded=cv2.erode(crop_roi, kernel);

    # For eliminating holes.
    eroded  = cv2.erode(crop_roi, kernel2);
    dilated = cv2.dilate(eroded, kernel2)

    #cv2.imshow("*Crop mask (after erose)", dilated)
    #cv2.waitKey(0)



    # Reference.
    #_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #bigestContour = myTools.contourSizeFiltering(contours)
    #cv2.drawContours(dilated, bigestContour, -1, 100, 10)

    #cv2.imshow("Crop mask (with contour)", dilated)
    #cv2.waitKey(0)

    # debug.
    #cv2.imshow("farmland_knnResult", farmland_knnResult)
    #cv2.imshow("farmland_knnResult red channel", farmland_redChannel_knnResult)
    return dilated


