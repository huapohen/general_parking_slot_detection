# --------------------------------------------------------
# process ps2.0_convert(tongji) and PIL(seoul) parking slots dataset
# based on this new dataset label, we can acquire pairable points or lines
# Written by liwenbo 2022-03
# --------------------------------------------------------
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


base_path = r'D:\dataset\pairable_parking_slot'

MarkingPoint = namedtuple(
    'MarkingPoint',  # 制作数据集不需要这个
    [   # 进入线的左边角点 （水平车向前行驶朝向的停车位的进入线是’长线‘，垂直的为‘短线’，斜的同样）
        'x',     # 取值范围：[0,H-1], [0,W-1]
        'y',     # 取值范围：
        'lenSepLine_x',    # delta_x 取值范围：归一化(-1, 1)
        'lenSepLine_y',    # delta_y 取值范围：归一化(-1, 1)
        # 'sinDiagLine',   # 加上对角就是任意多边形了，试一试这个版本
        # 选择水平线（x轴）为0度，
        # 右下为第一象限，左下为第二象限，左上为第三象限，右上为第四象限
        # pixel 坐标的x、y的增大方向为正向
        'lenEntryLine_x',  # delta_x 取值范围：归一化(-1, 1)
        'lenEntryLine_y',  # delta_y
        # 'lenSepLine',    # (delta_x, delta_y) 取值范围：归一化(-1, 1)
        # 一定是菱形
        # 'lenDiagLine',   # 加上这个，这就不一定
        # 'isOccupied',    # 取值范围：(0, 1) 这个loss系数量级要设低点，因为没啥用
        # 'slotType'])   # 这两个先不考虑，否则数据集还要重新制作
        # 不如制作下
        'isOccupied',   # 用sigmoid吧, // softmax多了一个channel，不想用
        # 'resolution',   # H,W
        # 'cmPerPixel',   # 实际物理距离 单位：厘米/像素
        # 'value_type',   # float(float64) 转 float32的时候要注意 ! 截断
    ])



# step 1
def file_rename(base_path):
    """
    delete top_subfile and bottom_subfile of tongji ps2.0_convert dataset
    and replace _l and _r to _left and _right
    Input:
        base_path = r'D:\dataset\pairable_parking_slot'
    Output:
    """
    print("\n ---- step 1: file_rename ---- \n")
    # setname = 'tongji'
    setname = 'fullimg'
    data_path = os.path.join(base_path, setname)
    for model in ['test', 'train']:
        for tag in ['image', 'label']:
            bp = os.path.join(data_path, model, tag)
            files = os.listdir(bp)
            print(f"process ps2.0_convert '{model}/{tag}' start: ")
            for f in tqdm(files):
                fp = os.path.join(bp, f)
                if "_b" in fp or "_t" in fp:
                    os.remove(fp)
            print(f"process ps2.0_convert '{model}/{tag}' finished ! \n")
        print("\n ---- step 1: finished ! ---- \n")
    return

# step 2
def label_clean(base_path):
    """
    delete the first two lines
    like:
        3   -> delete   I don't know what this stands for
        0   -> delete   , and this is also
    or:
        1   -> delete
        22  -> delete
        0 199 71 147 195 255 241 255 94
        the first digit represents whether the parking slot is occupied
        0 99 313 51 439 241 511 255 372
        0 147 195 99 313 255 376 255 239

    Input:
        (1) seoul PIL dataset labels path
        (2) tongji ps2.0_convert dataset labels path
        base_path = r'D:\dataset\pairable_parking_slot'
    Output:
    """
    print("\n ---- step 2: label_clean ---- \n")
    # for setname in ['seoul', 'tongji']:
    for setname in ['fullimg']:
        for mode in ['test', 'train']:
            bp = os.path.join(base_path, setname, mode, 'label')
            files = os.listdir(bp)
            print(f"process {setname} dataset '{mode}' label start: ")
            for file in tqdm(files):
                file = os.path.join(bp, file)
                with open(file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 1 or len(lines[0].split()) > 1:
                        # 防止误操作
                        break
                    # delete the first two line
                    lines = lines[2:]
                    # reserve isOccupied sign of parking slot
                    lines = [' '.join(line.split()) for line in lines]
                    str_lines = "\n".join(lines)
                with open(file, 'w+') as f:
                    f.write(str_lines)
            print(f"process {setname} dataset '{mode}' label finished ! \n")
        print("\n ---- step 2: finished ! ---- \n")
    return

# step 4
def prepare_json_format_tag(base_path, car_black_mask=50):
    print("\n ---- step 4: prepare_json_format_tag ---- \n")
    # for setname in ['seoul', 'tongji']:
    for setname in ['fullimg']:
        resolution = {}
        for mode in ['test', 'train']:
            print(f"prepare json format {setname} dataset '{mode}' start: ")
            suffix = '_align' if setname == 'seoul' else ''
            data_path = os.path.join(base_path, setname, mode)
            img_path = os.path.join(data_path, f'image{suffix}')
            lab_path = os.path.join(data_path, f'label{suffix}')
            img_path_final = os.path.join(data_path, 'image_final')
            lab_path_final = os.path.join(data_path, 'label_final')
            os.makedirs(img_path_final, exist_ok=True)
            os.makedirs(lab_path_final, exist_ok=True)
            # -------------------------------------
            for img_name in tqdm(os.listdir(img_path)):
                if '_r' in img_name:
                    continue
                lab_name = img_name.replace('jpg', 'txt')
                img_file = os.path.join(img_path, img_name)
                # lab_file = os.path.join(lab_path, lab_name)
                lab_file_l = os.path.join(lab_path, lab_name)
                lab_file_r = os.path.join(lab_path, lab_name.replace('_l', '_r'))
                ck_file = os.path.join(img_path_final, img_name)
                if os.path.isfile(ck_file):
                    # 防止误操作
                    break
                # -------------------------------------
                if 'for image':
                    i1 = cv2.imread(img_file)
                    i2 = cv2.imread(img_file.replace('_l', '_r'))
                    i1 = cv2.flip(i1, -1)
                    img = np.concatenate([i1, i2], axis=1)
                    H, W, C = img.shape
                    resolution['old'] = tuple([H, W])
                    resolution['new'] = tuple([512, 512])
                # -------------------------------------
                if 'for label':
                    isOccupied = []
                    x_list = []
                    y_list = []
                    for lab_file in [lab_file_l, lab_file_r]:
                        with open(lab_file, 'r') as f:
                            label = f.readlines()
                        for line in label:
                            if len(label) < 1:
                                break
                            line = [int(i) for i in line.split()]
                            coords = line[1:]
                            x, y = [], []
                            for i in range(len(coords)):
                                if i % 2:
                                    y.append(coords[i])
                                else:
                                    x.append(coords[i])
                            MINIMUM_COVER_RADIUS = 5
                            if pow((x[3]-x[0])**2 + (y[3]-y[0])**2, 0.5) < 3 \
                                    or pow((x[2]-x[1])**2 + (y[2]-y[1])**2, 0.5) \
                                    < MINIMUM_COVER_RADIUS:
                                # 这个点打得太紧了，要重合，跳过. 其实B、C可以不考虑
                                continue
                            isOccupied.append(line[0])
                            if '_l' in lab_file:
                                x = [256 - e for e in x]
                                y = [512 -e for e in y]
                            elif '_r' in lab_file:
                                x = [256 + e for e in x]
                            x_list.append(x)
                            y_list.append(y)
                # calculate finished !
                dicts = OrderedDict()
                dicts['isOccupied'] = isOccupied
                dicts['points_x']   = x_list
                dicts['points_y']   = y_list
                dicts['cmPerPixel'] = 1.67 # source from ps2.0
                dicts["subimgType"]  = setname
                dicts["datasetName"] = 'ps2.0'
                dicts["resolution"] = (512, 512)
                dicts["imgName"] = img_name.split('_l')[0]
                # -------------------------------------
                # save image & label
                lab_name = lab_name.replace('txt', 'json')
                img_file_final = os.path.join(img_path_final, img_name.replace('_l', ''))
                lab_file_final = os.path.join(lab_path_final, lab_name.replace('_l', ''))
                cv2.imwrite(img_file_final, img)
                with open(lab_file_final, 'w+') as f:
                    json.dump(dicts, f, indent=4)
        print("\n ---- step 4: finished ! ---- \n")

    return

# step 5
def rename_dir_and_rm_rawdata(base_path):
    # rename dir and rm previous dataset
    print("\n ---- step 4: rename_dir_and_rm_rawdata ---- \n")
    # for setname in ['seoul', 'tongji']:
    for setname in ['fullimg']:
        for mode in ['test', 'train']:
            print(f' ==== process {setname}/{mode} delete ==== ')
            cur_path = os.path.join(base_path, setname, mode)
            rename_list = []
            for dir in os.listdir(cur_path):
                dir = os.path.join(cur_path, dir)
                if '_final' not in dir:
                    # 只能执行一次，否则
                    print(f'rm {dir} ...')
                    shutil.rmtree(dir)
                else:
                    rename_list.append(dir)
            for dir in rename_list:
                new_name = dir.replace('_final', '')
                print(f'rename  {dir}')
                print(f'    to  {new_name} \n')
                os.rename(dir, new_name)
    print("\n ---- step 5: finished ! ---- \n")
    return

# step 6
def prepare_img_list_txt(base_path):
    ''' img_list.txt is used to quickly add, delete, check and modify '''
    print("\n ---- step 6: prepare_img_list_txt ---- \n")
    # for setname in ['seoul', 'tongji']:
    for setname in ['fullimg']:
        for mode in ['test', 'train']:
            print(f' ==== prepare {setname}/{mode}\t img_list.txt  ==== ')
            cur_path = os.path.join(base_path, setname, mode, 'image')
            txt_path = os.path.join(base_path, setname, mode, 'img_list.txt')
            file_list = os.listdir(cur_path)
            cnt_file = len(file_list)
            for i, file in enumerate(file_list):
                suff = '' if i == cnt_file - 1 else '\n'
                with open(txt_path, 'a') as f:
                    f.write(file.split('.')[0] + suff)
        break
    print("\n ---- step 6: finished ! ---- \n")
    return


if __name__ == '__main__':


    # base_path = r'D:\dataset\pairable_parking_slot'
    base_path = r'D:\dataset\parking_slot_dataset\2'
    if platform.system() == 'Linux':
        # base_path = r'/data/lwb/pairable_parking_slot'
        base_path = r'/home/data/lwb/data/ParkingSlot'

    file_rename(base_path)
    label_clean(base_path)
    prepare_json_format_tag(base_path)
    rename_dir_and_rm_rawdata(base_path)
    prepare_img_list_txt(base_path)


    print("\n\t #### DatasetMaker, completed ! #### \t\n")


