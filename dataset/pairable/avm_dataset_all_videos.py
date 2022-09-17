import os
import cv2
import sys
import glob
import json
import time
import math
import shutil
import platform
import numpy as np
from tqdm import tqdm
from enum import Enum
from collections import (namedtuple, OrderedDict)

for ind in range(12):
    bp = rf'D:\dataset\AVM\videos\20220424\extract_frames\{ind:04d}'
    avm_path = rf'D:\dataset\pairable_parking_slot\avm{ind:04d}\test'
    files = os.listdir(bp)
    for i in tqdm(files):
        name = i.split('.')[0]
        img_file = bp + rf'\{name}.png'
        img = cv2.imread(img_file)
        l_p = 432
        r_p = 648
        w = 432
        h = 432 * 3
        ratio = 432 / 128
        img_l = img[:,:432]
        img_r = img[:,648:]
        add_x = 108
        zeros = np.zeros([108, 432, 3], dtype=np.uint8) * 255
        img_l = np.array(np.concatenate([zeros, img_l, zeros], axis=0))
        img_r = np.array(np.concatenate([zeros, img_r, zeros], axis=0))
        img_l = cv2.resize(img_l, (128, 384), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(img_r, (128, 384), interpolation=cv2.INTER_AREA)
        img_l = img_l[::-1, ::-1]
        left_js = {"isOccupied": [], "points_x": [], "points_y": [],
                 "cmPerPixel": 0, "subimgType": "right",
                 "datasetName": "avm", "resolution": [384, 128],
                 "imgName": 'l_' + name
                 }
        right_js = {"isOccupied": [], "points_x": [], "points_y": [],
                 "cmPerPixel": 0, "subimgType": "right",
                 "datasetName": "avm", "resolution": [384, 128],
                 "imgName": 'r_' + name
                 }
        os.makedirs(avm_path + rf'\label', exist_ok=True)
        os.makedirs(avm_path + rf'\image', exist_ok=True)
        with open(avm_path + rf'\label\l_{name}.json', 'w+') as f:
            json.dump(left_js, f, indent=4)
        with open(avm_path + rf'\label\r_{name}.json', 'w+') as f:
            json.dump(right_js, f, indent=4)
        cv2.imwrite(avm_path + rf'\image\l_{name}.png', img_l)
        cv2.imwrite(avm_path + rf'\image\r_{name}.png', img_r)
        # img_list
        with open(avm_path + r'\img_list.txt', 'a') as f:
            f.write('l_' + name + '\n')
            f.write('r_' + name + '\n')



