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
import myTools

from sklearn import mixture



def func_getContours(shadow_mask):

    g_step_size = 20
    resultFile = "contourRst.png"
    rows = shadow_mask.shape[0]
    cols = shadow_mask.shape[1]

    # 1.dpgmm classification.
    X_ret2, Y_ret2 = myTools.build_trainSample(shadow_mask, g_step_size, False)
    X_train = np.array(X_ret2)

    n_components=20
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full').fit(X_train)



    # 2. draw contours.
    x    = np.linspace(0, rows)
    y    = np.linspace(0, cols)
    X, Y = np.meshgrid(x, y)
    XX   = np.array([X.ravel(), Y.ravel()]).T
    Z    = -dpgmm.score_samples(XX)
    Z    = Z.reshape(X.shape)

    figure = plt.figure(figsize=(9, 9))
    ax = plt.subplot(1, 1, 1)

    #plt.scatter(X_train[:, 0], X_train[:, 1], .8)
    ax.contourf(X, Y, Z, cmap=plt.cm.bone, levels=np.logspace(0, 2, 50))
    #CB = plt.colorbar(CS, shrink=0.8, extend='both')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xticks(())
    ax.set_yticks(())

    figure.subplots_adjust(left=.00, right=1, top=1, bottom=.00)
    figure.set_size_inches(rows/100, cols/100)
    resultFile = "contourRst.png"
    plt.savefig(resultFile, dpi=100)
    plt.show()

    img = cv2.imread(resultFile)
    return img, X_train



def func_statContour(img):

    hist_rgb_list = myTools.func_colorHist(img, False)
    hist_redChannel = hist_rgb_list[2]

    hist_redChannel_flatten = hist_redChannel.flatten()
    red_value_list = list(hist_redChannel_flatten)
    print("Red_Value_List:", red_value_list)

    avalid_red_value_list = []
    for colorIdx, pixelCnt in enumerate(red_value_list):
        #Jeff: I don't know why there are some tiny numbers.
        if pixelCnt > 400:
            avalid_red_value_list.append(colorIdx)

    # Get red channel picture, rotate due to format of plot.
    _, _, red_ch = cv2.split(img)
    red_ch = red_ch[::-1]
    red_ch = red_ch.T

    # Return.
    #print("Debug: colorIdx list:", avalid_red_value_list)
    #cv2.imshow("Debug: red_ch", red_ch)
    #cv2.waitKey(0)

    return red_ch, avalid_red_value_list


'''
def func_findWrapContour(grayContourImg, avalid_red_value_list, X_train):

    print("param: color list:", avalid_red_value_list)
    debug_img = grayContourImg.copy()
    for (x, y) in X_train:
        debug_img[x,y] = 255
    cv2.imshow("param: singleChannel Image", debug_img)
    cv2.waitKey(0)


    for color_value in avalid_red_value_list:

        single_mask = func_getSingleColorMask(grayContourImg, color_value)

        cv2.imshow("Iterate: [1] contour of single mask.", single_mask)

        #
        # fillPoly: https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
        #
        _, contours, hierarchy = cv2.findContours(single_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bigestContour = contourSizeFiltering(contours)
        cv2.drawContours(single_mask, bigestContour, -1, 50, 5)

        cv2.imshow("Iterate: [2] bigest contour.", single_mask)

        # -------------------------------------------------------------

        ret_list = []
        err_cnt  = 0
        for (x,y) in X_train:
            # There may be a bug in this api.
            ret = cv2.pointPolygonTest(bigestContour[0], (x, y), False)
            if (ret > 0):
                single_mask[x,y] = 100
            else:
                single_mask[x,y] = 255
                err_cnt = err_cnt + 1
            ret_list.append(ret)

            print(x,y, ret)
            cv2.imshow("Iterate: [3] in contour?", single_mask)
            cv2.waitKey(0)



        print("ret = ", ret_list, err_cnt/len(X_train) )

        cv2.imshow("Iterate: [3] in contour?", single_mask)
        cv2.waitKey(0)
'''



def func_findWrapContour(grayContourImg, avalid_red_value_list, X_train):

    print("param: color list:", avalid_red_value_list)
    debug_img = grayContourImg.copy()
    for (x, y) in X_train:
        debug_img[x,y] = 255
    cv2.imshow("param: singleChannel Image", debug_img)
    cv2.waitKey(0)


    for color_value in avalid_red_value_list:

        single_mask = myTools.func_getSingleColorMask(grayContourImg, color_value)

        cv2.imshow("Iterate: [1] contour of single mask.", single_mask)

        #
        # fillPoly: https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
        #
        _, contours, hierarchy = cv2.findContours(single_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bigestContour = myTools.contourSizeFiltering(contours)
        cv2.drawContours(single_mask, bigestContour, -1, 50, 5)

        cv2.imshow("Iterate: [2] bigest contour.", single_mask)

        # -------------------------------------------------------------
        flag_color = 123
        for (x,y) in X_train:
            single_mask[x,y] = flag_color

        cv2.fillPoly(single_mask, pts=bigestContour, color=20)

        cv2.imshow("Iterate: [3] cover all?", single_mask)
        cv2.waitKey(0)

        cover_cnt = 0
        for (x,y) in X_train:
            if not single_mask[x,y] == flag_color:
                cover_cnt = cover_cnt + 1
        print("cover_cnt = ", cover_cnt, len(X_train))
        if cover_cnt/len(X_train) > .95:
            return bigestContour

    return []
