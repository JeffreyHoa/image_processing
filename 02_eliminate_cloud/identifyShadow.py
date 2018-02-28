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


def model_get_shadow_mask(farmland_roi):

    ################################################################################
    # Find the main color value, which is also shadow land.
    ################################################################################

    # 1.1 knn, k = 5
    farmland_knnResult = myTools.func_applyKnn(farmland_roi, 5)

    hist_rgb_list = myTools.func_colorHist(farmland_knnResult)

    # 1.2 Identify shadow only based on red channel
    hist_redChannel = hist_rgb_list[2]
    #print("Debug: hist_redChannel", hist_redChannel)
    hist_redChannel_flatten = hist_redChannel.flatten()
    red_value_list = list(hist_redChannel_flatten)
    crop_redChannel_value = red_value_list.index(max(red_value_list))
    #print("Debug: crop_redChannel_value", crop_redChannel_value)


    # Jeff: shadow color value should be between 0 and crop color.
    shadow_redChannel_value = 0
    isThereShadow = False
    for idx in range(1, crop_redChannel_value):
        if red_value_list[idx] > 0:
            isThereShadow = True
            shadow_redChannel_value = idx
            break

    if isThereShadow == False:
        print("Notice: There is no shadow.")
        _, _, emptyImage = cv2.split(farmland_knnResult)
        ret, emptyImage = cv2.threshold(emptyImage, 0, 0, cv2.THRESH_BINARY)

        return emptyImage, False


    ################################################################################
    # Get shadow land mask.
    ################################################################################

    # 2.1 Get red channel.
    _, _, farmland_redChannel_knnResult = cv2.split(farmland_knnResult)


    # 2.2 Get mask for shadow.
    shadow_roi = farmland_redChannel_knnResult.copy()

    for row_idx in range(len(farmland_redChannel_knnResult)):
        for col_idx in range(len(farmland_redChannel_knnResult[row_idx])):
            if not farmland_redChannel_knnResult[row_idx][col_idx] == shadow_redChannel_value:
                shadow_roi[row_idx][col_idx] = 0
            else:
                shadow_roi[row_idx][col_idx] = 255


    #cv2.imshow("shadow ROI.", shadow_roi)
    #cv2.waitKey(0)



    ################################################################################
    # Erose land mask for better performance.
    ###############################################################################n

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(30, 30))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    kernel3  = cv2.getStructuringElement(cv2.MORPH_RECT,(40, 40))

    # For robust
    #eroded=cv2.dilate(shadow_roi, kernel2);

    # For eliminating holes.
    eroded=cv2.erode(shadow_roi, kernel);
    dilated = cv2.dilate(eroded, kernel)

    #cv2.imshow("shadow mask (after erose)", dilated)
    #cv2.waitKey(0)



    # Reference.
    #_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #bigestContour = myTools.contourSizeFiltering(contours)
    #cv2.drawContours(dilated, bigestContour, -1, 100, 10)

    #cv2.imshow("shadow mask (with contour)", dilated)
    #cv2.waitKey(0)

    return dilated, True


