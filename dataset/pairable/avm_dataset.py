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


bp = rf'D:\dataset\AVM\videos\extract_frames_remark\avm_marking_merge'
avm_path = r'D:\dataset\AVM\videos\avm\train'
files = os.listdir(bp)
for i in tqdm(files):
    name = i.split('.')[0]
    img_file = bp + rf'\{name}.png'
    json_file = bp + rf'\{name}.json'
    img = cv2.imread(img_file)
    with open(json_file, 'r') as f:
        js = json.load(f)
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
    cnt_ps = len(js['shapes'])
    if cnt_ps % 2 != 0:
        print(img_file)
        continue
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
    p_dict = {}
    ps_org = js['shapes']
    for i in range(cnt_ps):
        gp = ps_org[i]['group_id']
        if gp not in p_dict:
            p_dict[gp] = ps_org[i]
        else:
            p_dict[gp]['points'].extend(ps_org[i]['points'])
            tmp_ps = p_dict[gp]['points']
            cnt_r, cnt_l = 0, 0
            for j in range(len(tmp_ps)):
                if tmp_ps[j][0] >= 648:
                    cnt_r += 1
                if tmp_ps[j][0] < 432:
                    cnt_l += 1
            if cnt_r == 4:
                # right
                right_js['isOccupied'].append(int(p_dict[gp]['free_flag']))
                tmp_x = []
                tmp_y = []
                for k in range(4):
                    org_x = (tmp_ps[k][0] - 648) / ratio
                    org_y = (tmp_ps[k][1] + 108) / ratio
                    if k == 2:
                        org_x = (tmp_ps[3][0] - 648) / ratio
                        org_y = (tmp_ps[1][1] + 108) / ratio
                    tmp_x.append(org_x)
                    tmp_y.append(org_y)
                right_js['points_x'].append(tmp_x)
                right_js['points_y'].append(tmp_y)
                right_js['subimgType'] = 'right'
            elif cnt_l == 4:
                # left
                left_js['isOccupied'].append(int(p_dict[gp]['free_flag']))
                tmp_x = []
                tmp_y = []
                for k in range(4):
                    org_x = tmp_ps[k][0] / ratio
                    org_y = (tmp_ps[k][1] + 108) / ratio
                    if k == 2:
                        org_x = tmp_ps[3][0] / ratio
                        org_y = (tmp_ps[1][1] + 108) / ratio
                    tmp_x.append(127 - org_x)
                    tmp_y.append(383 - org_y)
                left_js['points_x'].append(tmp_x)
                left_js['points_y'].append(tmp_y)
                left_js['subimgType'] = 'left'
            else:
                continue
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



