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
# Create folder ./data and move photos in it.
# Then, $ python main.py
# Finally, the name of result starts with "square" in ./data
# =============================================================================

import cv2
from os import listdir
from os.path import isfile, join

import os
import sys
#import re

################################################################################
# Load all files then regex
################################################################################
curPath = os.getcwd()
dirPath = join(curPath, "data")
if not os.path.isdir(dirPath):
    print("Warn: There is no './data' here")
    sys.exit()

filteredFiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
if len(filteredFiles) == 0:
    print("Warn: There is no photo in folder: ./data")
    sys.exit()

print("Files you want:")
print(filteredFiles, "\n")


################################################################################
# Jeff: find find the final size.
################################################################################
numList = []

for name in filteredFiles:
    img  = cv2.imread(join(dirPath, name))
    rows = img.shape[0]
    cols = img.shape[1]
    numList.append(rows)
    numList.append(cols)

print("Debug:",numList)

final_size = min(numList)
print("Min num (final size):", final_size, "\n")


################################################################################
# Jeff: find out the max square shape.
################################################################################

for name in filteredFiles:
    img  = cv2.imread(join(dirPath, name))
    print("-----------------------")
    print("Debug:", name, img.shape)
    rows = img.shape[0]
    cols = img.shape[1]

    start_row = 0
    start_col = 0
    if (rows > cols):
        # vertical
        start_row = (rows - cols) // 2
    else:
	# horizon
        start_col = (cols - rows) // 2
    min_edge=min(rows, cols)


    imgROI=img[start_row:start_row+min_edge, start_col:start_col+min_edge]


    print("Debug:", imgROI.shape)
#    cv2.imshow("show", imgROI)
#    cv2.waitKey(0)

    img_zo = cv2.resize(imgROI, (final_size, final_size), interpolation=cv2.INTER_AREA)

    print("Debug:", img_zo.shape)
#    cv2.imshow("show", img_zo)
#    cv2.waitKey(0)

    ret_name = "square_"+name
    cv2.imwrite(join(dirPath, ret_name), img_zo, params=None)


