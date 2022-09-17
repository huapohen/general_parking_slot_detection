import os
import sys
import json
import shutil
import tqdm
from glob import glob



bp = r'D:\dataset\pairable_parking_slot'
for ds in ['seoul', 'tongji']:
    ds_mode_dict = {}
    ds_mode_pA = 1
    for mode in ['train', 'test']:
        mode_pA = 1
        files_path = bp + rf'\{ds}\{mode}\label'
        files = os.listdir(files_path)
        for file in files:
            path = files_path + rf'\{file}'
            with open(path, 'r') as f:
                label = json.load(f)
            H, W = label['resolution']
            gap = 1
            if len(label['points_x']) == 0:
                continue
            for i,p1 in enumerate(label['points_x']):
                for j,p2 in enumerate(label['points_x']):
                    yA = label['points_y'][i]
                    yB = label['points_y'][j]
                    if j > i:
                        gap = min(pow((p2[0]/W - p1[0]/W)**2 + \
                                     (yA[0]/H - yB[0]/H)**2, 0.5), gap)
            mode_pA = min(mode_pA, gap)
        print(ds, mode, mode_pA)
        ds_mode_pA = min(ds_mode_pB, mode_pA)
    ds_mode_dict[ds + '_point_A'] = ds_mode_pA

'''
seoul train 0.08571850213558112
seoul test 0.10736474448038716
tongji train 0.21255609471398018
tongji test 0.21442495126705652
'''