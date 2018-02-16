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
from os import listdir
from os.path import isfile, join

import os
import re

################################################################################
# Load all files then regex
################################################################################

def func_loadData(dirPath, suffix):

    if not os.path.isdir(dirPath):
        print("Warn: There is no './data' here")
        return []

    regex = ".*\\" + suffix
    filteredFilePathList = [join(dirPath,f) for f in listdir(dirPath) if isfile(join(dirPath, f)) and re.search(regex,f)]
    if len(filteredFilePathList) == 0:
        print("Warn: There is no .tif in folder: ./data")
        return []

    return filteredFilePathList


################################################################################
# Check the size of .tif
################################################################################

def func_isAvailiableData(filePathList):
    rows = 0
    cols = 0

    for filePath in filePathList:

        # 1.
        img  = cv2.imread(filePath)
        print("-----------------------")
        print("Debug:", filePath, img.shape)

        # 2.
        if (rows == 0 or cols == 0):
            rows = img.shape[0]
            cols = img.shape[1]
        else:
            pic_rows = img.shape[0]
            pic_cols = img.shape[1]

            if (pic_rows == rows and pic_cols == cols):
                pass;
            else:
                print("Error: size of .tif is not constant.")
                return False
    return True


