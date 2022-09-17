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


ind = 11
print(f'{ind:04d}')
bp = rf'D:\dataset\AVM\videos\20220424\extract_frames\{ind:04d}'
files = os.listdir(bp)
pd_path = r'D:\code\gitlab\deploy\experiments\parking_slot_detection\exp_120\image\pd\avm'
merge_path = rf'D:\dataset\AVM\videos\avm\merge_all_videos_2\{ind:04d}'
os.makedirs(rf'D:\dataset\AVM\videos\avm\merge_all_videos_2\{ind:04d}', exist_ok=True)
k = 0
fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
output_path = merge_path + '.mp4'
writer = cv2.VideoWriter(output_path, fourcc, 10, (320*2+30, 320))
for i in tqdm(files):
    name = i.split('.')[0]
    img_file = bp + rf'\{name}.png'
    img = cv2.imread(img_file)
    l_p = 432
    r_p = 648
    w = 432
    h = 1296
    ratio = 432 / 128
    add_x = 108
    org_img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
    zeros = np.zeros([108, 1080, 3], dtype=np.uint8) * 255
    img = np.array(np.concatenate([zeros, img, zeros], axis=0))
    img_m = img[:, 432:648]
    img_m = cv2.resize(img_m, (64, 384), interpolation=cv2.INTER_AREA)
    img_l_path = pd_path + r'\l_' + name + '.jpg'
    img_r_path = pd_path + r'\r_' + name + '.jpg'
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    img_l = img_l[::-1, ::-1]
    img_merge = np.array(np.concatenate([img_l, img_m, img_r], axis=1))
    img_merge = img_merge[32:352]
    ones = np.ones([320, 30, 3], dtype=np.uint8) * 255
    img_merge = np.array(np.concatenate([org_img, ones, img_merge], axis=1))
    writer.write(img_merge)
    cv2.imwrite(merge_path + rf'\{name}.jpg', img_merge)
    k += 1
writer.release()


if 0:
    fps = 5
    check_call(['ffmpeg',
                '-r', str(fps),
                # '-v', 'quiet',
                '-i', r'D:\dataset\AVM\videos\avm\merge\%05d.jpg',
                '-vcodec', 'mpeg4',
                '-b:v', '6000k',
                rf'D:\dataset\AVM\videos\avm\fps_{fps}.avi'],
               shell=True)
'''

ffmpeg -r  15 -i D:\dataset\AVM\videos\avm\merge\%05d.jpg -vcodec mpeg4 -b:v 6000k D:\dataset\AVM\videos\avm\fps_15.avi
ffmpeg -r  5 -i D:\dataset\AVM\videos\avm\merge\%05d.jpg -vcodec mpeg4 -b:v 6000k D:\dataset\AVM\videos\avm\fps_5.avi
ffmpeg -r  10 -i D:\dataset\AVM\videos\avm\merge\%05d.jpg -vcodec mpeg4 -b:v 6000k D:\dataset\AVM\videos\avm\fps_10.avi

'''



