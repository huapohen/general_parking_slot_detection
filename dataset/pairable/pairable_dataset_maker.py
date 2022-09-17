# --------------------------------------------------------
# process ps2.0_convert(tongji) and PIL(seoul) parking slots dataset
# based on this new dataset label, we can acquire pairable points or lines
# --------------------------------------------------------
import os
import cv2
import sys
import glob
import json
import time
import math
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
        'cosEntryLine',  # 取值范围: (-1, 1) tanh 没有端点
        'sinSepLine',    #
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
        # 'isOccupied',   # 用sigmoid吧, // softmax多了一个channel，不想用
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
    setname = 'tongji'
    data_path = os.path.join(base_path, setname)
    for model in ['test', 'train']:
        for tag in ['image', 'label']:
            bp = os.path.join(data_path, model, tag)
            files = os.listdir(bp)
            print(f"process ps2.0_convert '{model}/{tag}' start: ")
            for f in tqdm(files):
                fp = os.path.join(bp, f)
                if '_right' in fp or '_left' in fp:
                    continue
                if "_b" in fp or "_t" in fp:
                    os.remove(fp)
                if "_l" in fp:
                    new_fp = fp.replace("_l", "_left")
                    os.rename(fp, new_fp)
                if "_r" in fp:
                    new_fp = fp.replace("_r", "_right")
                    os.rename(fp, new_fp)
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
    for setname in ['seoul', 'tongji']:
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

# step 3
def pixel_centimetre_align(base_path, pixcm1=1.875, pixcm2=1.67):
    """
    It can be known from ps2.0 dataset paper and PIL dataset paper
    that the distance represented by the pixels is inconsistent.
    For ps2.0, 1 pixel on the AVM image corresponds to 1.67cm,
    but for PIL it is 1.875cm.
    For ps2.0, the resolution is 600x600, and for PIL it is 768x256
    You should zoom in the denser image: PIL
    resolution: (768,256,3) -> (862,287,3) -> (864, 288) -> /32 = (27, 9)
    zoomin_ratio: 1.875 / 1.67 = 1.122754491017964
    Input:
        base_path
        dist_1
        dist_2
    Ouput: used for cut image
    """
    print("\n ---- step 3: pixel_centimetre_align ---- \n")
    zoomin_ratio = pixcm1 / pixcm2 # 1.122754491017964
    setname = 'seoul'
    data_path = os.path.join(base_path, setname)
    resolution = {}
    for mode in ['test', 'train']:
        print(f"zoomin {setname} dataset '{mode}' start: ")
        img_path = os.path.join(data_path, mode, 'image')
        lab_path = os.path.join(data_path, mode, 'label')
        img_path_align = os.path.join(data_path, mode, 'image_align')
        lab_path_align = os.path.join(data_path, mode, 'label_align')
        os.makedirs(img_path_align, exist_ok=True)
        os.makedirs(lab_path_align, exist_ok=True)
        # zoomin PIL(seoul) dataset
        for img_name in tqdm(os.listdir(img_path)):
            lab_name = img_name.replace('jpg', 'txt')
            img_file = os.path.join(img_path, img_name)
            lab_file = os.path.join(lab_path, lab_name)
            ck_file = os.path.join(img_path_align, img_name)
            if os.path.isfile(ck_file):
                # 防止误操作
                return
            # -------------------------------------
            if 'for image':
            # if 0: # for skip
                img = cv2.imread(img_file)
                resolution['old'] = img.shape[:2]
                img_align = cv2.resize(img, dsize=None,
                                       fx=zoomin_ratio, fy=zoomin_ratio,
                                       interpolation=cv2.INTER_CUBIC)
                # (768,256,3) -> (862,287,3) -> (864, 288) -> /32 = (27, 9)
                # 有必要对齐吗？ 算了，不要对齐
                # 看下实测效果再说  (    )
                # Need ToDo FixME
                # 反正卡阈值不得行了，卡阈值画槽瞬间失效
                H, W, C = img_align.shape
                H_align = round(H / 32 + 0.5) * 32
                W_align = round(W / 32 + 0.5) * 32
                resolution['new'] = tuple([H_align, W_align])
                img_align = cv2.copyMakeBorder(
                    img_align,
                    top=0, bottom=H_align - H,
                    left=0, right=W_align - W,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0)
                img_file_align = os.path.join(img_path_align, img_name)
                cv2.imwrite(img_file_align, img_align)
            # -------------------------------------
            if 'for label':
            # if 0: # for skip
                with open(lab_file, 'r') as f:
                    label = f.readlines()
                str_label_align = ""
                for line in label:
                    line_align = []
                    for k, v in enumerate(line.split()):
                        # k=0 represents isOccupied sign
                        if k > 0: # x,y -> W,H
                            v = min(round(int(v) * zoomin_ratio + 0.5),
                                    W_align if k % 2 else H_align)
                        line_align.append(v)
                    line_align = [str(i) for i in line_align]
                    str_label_align += " ".join(line_align) + "\n" # 换行
                lab_file_align = os.path.join(lab_path_align, lab_name)
                with open(lab_file_align, 'w+') as f:
                    f.write(str_label_align)
        #
        print(f"zoomin {setname} dataset '{mode}' finished! \n" +
              f"zoomin_ratio: {pixcm1} / {pixcm2} = {zoomin_ratio}\n" +
              f" origin_resolution_HxW = ({H}, {W})\n" +
              f"aligned_resolution_HxW = ({H_align}, {W_align})\n"
              )
        print("\n ---- step 3: finished ! ---- \n")
    return

# step 4
def prepare_json_format_tag(base_path, car_black_mask=50):
    """
    All datasets should contain the following types of labels:
        reference definition of MarkingPoint(namedtuple)
        The definition is at the beginning of this program. above ↑↑
    Convert label.txt to label.json with namedtuple format.
    label.txt -> label.json
    seoul（PIL）的数据集制作方式是：
    从车位进入线的左边角点开始逆时针旋转，标记车位的检测框需要的4个点。
    前两个为进入线的左右角点，后两个为组成矩形框所需要的点，不代表车位角点。
    车位可以这样分三种类型来解释这个seoul的标法：

       A.____________       D.____________     H.____________
        |           |        |           |      |           |
      ==>    1      |      ==>     3     |    ==>     5     |
       B.___________|       E|___________|     I|____.______|_____
        |           |       F.____________           |           |
      ==>    2      |        |           |         ==>     6     |
       C|___________|      ==>     4     |          J|___________|
                            G|___________|
        图（1）并线车位        图（2）间隔车位        图（3）并线错位车位

        不管是斜的车位、还是矩形、或者任意形状的车位，也不管是垂直车位还是水平车位，
    都可以按上图这样分为三种。其中，第（1）种和第（3）种为并线车位，细分的错位于不错位；
    若是间隔车位，只管自己本身这个车位，周围任何其他车位都不用考虑。
    上述是两两关系，若出现第3个及以上车位，按上述分类法则，可继续细分类。
    这样分类，主要是利用————“目标角点”。
        ==> 代表entry_line的方向，水平、垂直或斜车位都按这样规定来进入。
        .   代表这是个目标角点
        字母 代表点
        数字 代表停车位编号
        如图（1）所示，有3个点：A、B、C，其中A点为“1”车位的目标角点；
    B为“2”车位的目标角点；C为“2”车位的entry_line（进入线）的辅助角点；
    这里要注意的是，B既是“1”车位的辅助角点，同时也是“2”车位的目标角点。
    并线的循环标法，把角点利用起来了。
        隔间车位不用关心辅助角点和目标角点共点的关系。
        并线错位车位不用关心辅助角点和目标角点共点的关系。
    示例：
            A.____________D
             |           |
          ==>     1      |
            B.___________|C
             |           |
          ==>     2      |
            E|___________|F

        “1”车位：A(175, 41) B(170,183) C(255,186) D(255, 44)
        “2”车位：B(170,183) E(166,323) F(255,325) C(255,185)
        点的位置为逆时针旋转
        B点为(170,183)，共用了。
        另外，有个画框的点此时也共点了(255,185)，不考虑。 185vs186
        画框的点标签都打得不一样了，更不用考虑，
        倒是要注意下共用的目标角点，需要判断下，
        必要时进行修复，如abs(pixel_delta) < 1
    注意：
        根据上述分析，并线有时并不是“真的”并线（坐标完全吻合），
        这只是一种人为的分法，方便设计
    后记：
        十分依赖seoul(PIL)的标注质量 !
        后续需要人工校验，写代码测试（   ）ToDo
    Input:
        base_path
        car_black_mask: 50 依照tongji数据集的左、右子图mask区域像素的宽度
            人为设定值
    Output:
    """
    print("\n ---- step 4: prepare_json_format_tag ---- \n")
    for setname in ['seoul', 'tongji']:
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
                lab_name = img_name.replace('jpg', 'txt')
                img_file = os.path.join(img_path, img_name)
                lab_file = os.path.join(lab_path, lab_name)
                ck_file = os.path.join(img_path_final, img_name)
                if os.path.isfile(ck_file):
                    # 防止误操作
                    break
                # -------------------------------------
                if 'for image':
                    img = cv2.imread(img_file)
                    H, W, C = img.shape
                    resolution['old'] = tuple([H, W])
                    # 放大填充，不要减小（否则会丢掉像素） tongji
                    H_final = int(H / 3 + 0.5) * 3  # 512 -> 513
                    W_final = int(H / 3 + 0.5)      # 256 -> 171
                    resolution['new'] = tuple([H_final, W_final])
                    H_pool = H_final - H if H_final > H else 0
                    W_pool = W_final - W if W_final > W else 0
                    if setname == 'tongji':
                        img = cv2.copyMakeBorder(
                            img,
                            top=0, bottom=H_pool,
                            left=0, right=W_pool,
                            borderType=cv2.BORDER_CONSTANT,
                            value=0)
                        # 切图
                        img = img[:, car_black_mask:(W_final+car_black_mask):, :]
                # -------------------------------------
                if 'for label':
                    objectPoint  = []
                    cosEntryLine = []
                    sinSepLine   = []
                    sinDiagLine  = []
                    lenEntryLine = []
                    lenSepLine   = []
                    lenDiagLine  = []
                    isOccupied = []
                    with open(lab_file, 'r') as f:
                        label = f.readlines()
                    for line in label:
                        if len(label) < 1:
                            break
                        line = [int(i) for i in line.split()]
                        # # 车位是否可用   1：可用，0：不可用
                        # isOccupied.append(line[0]) # 放到下面去写
                        # 取坐标
                        coords = line[1:]
                        # 割图，seoul的数据集，有太靠近边缘的点，不能割，填充可以考虑
                        # seoul 和 tongji 数据集的纵横比例割得一样
                        # tongji 割黑色区域吧，默认；
                        # 顶部和底部会被割掉，上子图不好做
                        # tongji 的中间黑色区域，子图割50像素，纵横满足seoul 3:1关系
                        # 对边割: 256 - 50 - int(512/3+0.5)，
                        # 放大填充，不要减小（会丢掉像素）
                        x, y = [], []
                        for i in range(len(coords)):
                            if i % 2:
                                y.append(coords[i])
                            else:
                                x.append(coords[i])
                        if setname == 'tongji':
                            # (512,256) -> (513,171) 3:1
                            # 顶、底 部的点，可能有误差
                            x = [min(max(i-car_black_mask, 0), W_final-1) for i in x]
                        # 两个标点的欧式距离不能小于这个数
                        MINIMUM_COVER_RADIUS = 5
                        if pow((x[3]-x[0])**2 + (y[3]-y[0])**2, 0.5) < 3 \
                                or pow((x[2]-x[1])**2 + (y[2]-y[1])**2, 0.5) \
                                < MINIMUM_COVER_RADIUS:
                            # 这个点打得太紧了，要重合，跳过. 其实B、C可以不考虑
                            continue
                        # 存目标点
                        objectPoint.append([x[0], y[0]])
                        # 存 线的方向角度，水平线选择为0度，逆时针+、顺时针-
                        # tanh (-1, 1) 没有端点值
                        x0,x1,x2,x3 = x
                        y0,y1,y2,y3 = y
                        cos_AB = (x1-x0) / pow((x1-x0)**2 + (y1-y0)**2, 0.5)
                        sin_AD = (y3-y0) / pow((x3-x0)**2 + (y3-y0)**2, 0.5)
                        sin_AC = (x2-x0) / pow((x2-x0)**2 + (y2-y0)**2, 0.5)
                        # tanh 网络预测(x0,y0)和_len，由(h,w)可推出x1x2x3,y1y2y3
                        entry_len = [(x1 - x0) / W_final, (y1 - y0) / H_final]
                        sep_len = [(x3 - x0) / W_final, (y3 - y0) / H_final]
                        diag_len = [(x2 - x0) / W_final, (y2 - y0) / H_final]
                        # 存
                        isOccupied.append(line[0]) # 车位
                        cosEntryLine.append(cos_AB)
                        sinSepLine.append(sin_AD)
                        sinDiagLine.append(sin_AC)
                        lenEntryLine.append(entry_len)
                        lenSepLine.append(sep_len)
                        lenDiagLine.append(diag_len)
                        value_type = str(type(sin_AD))
                        # 注意 是 float多少值,prt看一哈，
                        # float64(float)和float32有差别
                # calculate finished !
                dicts = OrderedDict()
                # (x, y): [0, 1]
                dicts["objectPoint"]  = objectPoint
                dicts['cosEntryLine'] = cosEntryLine
                dicts['sinSepLine']   = sinSepLine
                dicts['sinDiagLine']  = sinDiagLine
                dicts["lenEntryLine"] = lenEntryLine
                dicts["lenSepLine"]   = lenSepLine
                dicts['lenDiagLine']  = lenDiagLine
                # 0 or 1
                dicts["isOccupied"]   = isOccupied
                dicts["cmPerPixel"]   = 1.67 # source from ps2.0
                # 'left' or 'right' or 'top' or 'bottom'
                # 都往右边转，左子图镜像flip到右，上子图顺时针到右，下子图逆时针到右
                # 维持右子图左上角点逆时针的旋转顺序
                # seoul的处理是左子图原地旋转180度到右，旋转后车位顺序从图中是从下到上数，注意！
                # 上子图旋转会出现奇怪的车位，即左边的停车位旋转后跑到上边儿去了，entry_line变横了
                # ，那么左子图和右子图都会处理上边儿的车位
                dicts["subimgType"]  = 'right' if setname in 'seoul' \
                    else img_name.split('_')[-1].split('.')[0] # seoul全右图
                # 'seoul' or 'tongji'
                dicts["datasetName"] = setname
                dicts["resolution"] = (H_final, W_final)
                dicts["imgName"] = img_name.split('.')[0]
                # dicts["value_type"] = value_type
                # -------------------------------------
                # save image & label
                lab_name = lab_name.replace('txt', 'json')
                img_file_final = os.path.join(img_path_final, img_name)
                lab_file_final = os.path.join(lab_path_final, lab_name)
                # if setname == 'tongji': # seoul也存，方便管理
                cv2.imwrite(img_file_final, img)
                with open(lab_file_final, 'w+') as f:
                    json.dump(dicts, f, indent=4)
            # print(f"prepare json format {setname} dataset '{mode}' finished! \n" +
            #       f" origin_resolution_HxW = ({H}, {W})\n" +
            #       f"aligned_resolution_HxW = ({H_final}, {W_final})\n"
            #       )
        print("\n ---- step 4: finished ! ---- \n")

    return


def rename_dir(base_path):
    import shutil
    for setname in ['seoul', 'tongji']:
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
    return


if __name__ == '__main__':

    # base_path = r'D:\dataset\pairable_parking_slot'
    base_path = r'/data/lwb/pairable_parking_slot'
    if 0:
    # if 1:
        print('''\n\t #### This is DatasetMaker for pairable marking point. #### \n''')
        time.sleep(3)
        print(rf'''
         The structure of input directory should like as:
            {base_path}\tongji\train\image\20160725-3-1_left.jpg
            {base_path}\tongji\train\label\20160725-3-1_left.txt
            {base_path}\seoul\test\image\*
            {base_path}\seoul\test\label\*
            ...
        ''')
        time.sleep(3)
        # ---------------------------------------
        #           following the steps
        # ---------------------------------------
        p0 = f'\n\t Create a dataset in the following order:'
        p1 = f'\t\t step 1:  file_rename'
        p2 = f'\t\t step 2:  label_clean'
        p3 = f'\t\t step 3:  pixel_centimetre_align'
        p4 = f'\t\t step 4:  prepare_json_format_tag'
        for i in [p0, p1, p2, p3, p4]:
            print(i)
            time.sleep(1)
        print(f'\t Start ! {time.ctime()}')
        time.sleep(0.2)
        # step 1
        file_rename(base_path)
        # step 2
        label_clean(base_path)
        # step 3
        pixel_centimetre_align(base_path)
        # step 4
        prepare_json_format_tag(base_path)



    print("\n\t #### DatasetMaker, completed ! #### \t\n")

    # # rename dir and rm previous dataset
    # rename_dir(base_path)
