"""Defines the parking slot dataset for directional marking point detection."""
import logging
from math import e
import os
import sys
import random
import cv2
import json
from glob import glob
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from dataset.process import *

_logger = logging.getLogger(__name__)


class AutoParkingData(Dataset):
    def __init__(self, params, mode="train"):
        # 参数预设
        self.params = params
        if mode == "train":
            self.data_ratio = params.train_data_ratio
            self.mode = 'train'
        else:
            self.data_ratio = params.val_data_ratio
            self.mode = 'test'
        self.set_path = params.data_dir
        self.is_deploy = params.is_BNC
        random.seed(params.seed)
        self.enable_random = params.enable_random
        self.sample_infor = []
        # ['seoul', 'tongji', 'avm']
        for data_ratio in self.data_ratio:
            setname = data_ratio[0]
            self.img_type = 'jpg' if setname in ['seoul', 'tongji'] else 'png'
            img_list_path = os.path.join(self.set_path, setname, mode,
                                         'img_list.txt')
            if os.path.exists(img_list_path):
                with open(img_list_path, 'r') as f:
                    imgs_file = f.readlines()
                    tmp_path = os.path.join(self.set_path, setname, mode, 'image')
                    imgs_file = [os.path.join(tmp_path,
                                              k.split('\n')[0] + '.' + self.img_type)
                                 for k in imgs_file]
            else:
                data_path = os.path.join(self.set_path, setname, mode,
                                         'image', '*')
                imgs_file = glob(data_path)
            if len(imgs_file) == 0: # for avm test
                imgs_file = glob(os.path.join(self.set_path, '*'))
            # sample dataset
            percentage = int(len(imgs_file) * data_ratio[1])
            if self.enable_random:
                imgs_file = random.sample(imgs_file, percentage)
            else:
                imgs_file = [i for i in imgs_file[:percentage]]
            self.sample_infor += imgs_file
        # shuffle
        if self.enable_random:
            random.shuffle(self.sample_infor)

    def __getitem__(self, index):
        image_file = self.sample_infor[index]
        self.index = index
        self.image_file = image_file
        image = cv2.imread(image_file)
        tmp_list = image_file.split(os.sep)
        tmp_list = ['label' if i == 'image' else i for i in tmp_list]
        label_file = (os.sep).join(tmp_list)
        label_file = label_file.replace(self.img_type, 'json')
        try:
            with open(label_file, 'r') as f:
                label = json.load(f)
        except:  # for avm test
            img_name = image_file.split(os.sep)[-1].split('.')[0]
            label = {
                "isOccupied": [],
                "points_x": [],
                "points_y": [],
                "cmPerPixel": 1.67,
                "subimgType": "right",
                "datasetName": "avm",
                "resolution": [
                    864,
                    288
                ],
                "imgName": f"{img_name}"
            }
        is_gray = self.params.in_dim == 1
        image, marking_point = self.data_aug(image, label, is_gray)
        label["image"] = image
        label["MarkingPoint"] = marking_point
        return label

    def data_aug(self, image, json_data, gray=False):
        def augment_hsv(img, hsv_prob=1, hgain=5, sgain=30, vgain=30):
            """modify Hue, Saturation and Value of image randomly"""
            if np.random.rand() < hsv_prob:
                hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain
                                                          ]  # random gains
                hsv_augs *= np.random.randint(0, 2,
                                              3)  # random selection of h, s, v
                hsv_augs = hsv_augs.astype(np.int16)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

                img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
                img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0,
                                          255)
                img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0,
                                          255)
                # no return needed
                cv2.cvtColor(img_hsv.astype(img.dtype),
                             cv2.COLOR_HSV2BGR,
                             dst=img)

        def rotate_image(image, angle_degree):
            """Rotate image with given angle in degree."""
            h, w, _ = image.shape
            rotation_matrix = \
                cv2.getRotationMatrix2D((w / 2, h / 2),
                                        angle_degree, 1)
            # warpAffine  会有黑色区域出现
            return cv2.warpAffine(image, rotation_matrix, (w, h))

        def rotate_vector(vector, angle_degree):
            """Rotate a vector with given angle in degree."""
            angle_rad = math.pi * angle_degree / 180
            xval = vector[0]*math.cos(angle_rad) + \
                vector[1]*math.sin(angle_rad)
            yval = -vector[0]*math.sin(angle_rad) + \
                vector[1]*math.cos(angle_rad)
            return xval, yval

        def boundary_check(points, params):
            """Check situation that marking point appears too near to border."""
            margin = [params.input_size[0] / 20,
                      params.input_size[1] / 20]  # (384/20, 128/20)
            for point in points:
                # 每一条停车线单独遍历，若角点距离边缘过近则不作为训练集使用
                if point[0] < margin[1] or point[0] > (params.input_size[1]-margin[1]) \
                        or point[1] < margin[0] or point[1] > (params.input_size[0]-margin[0]):
                    return False
            return True

        def overlap_check(points, params):  # 这种情况似乎应该不多见？
            """Check situation that multiple marking points appear in same cell."""
            for i in range(len(points) - 1):
                i_x = points[i, 0]
                i_y = points[i, 1]
                for j in range(i + 1, len(points)):
                    j_x = points[j, 0]
                    j_y = points[j, 1]
                    if abs(j_x - i_x) < params.input_size[1] / params.feature_map_size[1] \
                            and abs(j_y - i_y) < params.input_size[0] / params.feature_map_size[0]:
                        return True  # overlap了，返回True
            return False  # 没有overlap

        def rotate_data(image, json_data, params):
            # 这一句为什么会这么影响效果？
            px_ori = np.array(json_data["points_x"]).astype(np.float32)
            py_ori = np.array(json_data["points_y"]).astype(np.float32)

            if random.random() < (1 - params.rotate_p):
                return image, json_data

            angle_range = params.angle_range
            angle_degree = random.randrange(-angle_range, angle_range)
            angle_degree = (360 - angle_degree) \
                if angle_degree < 0 else angle_degree

            h, w, c = image.shape
            if len(json_data["points_x"]) == 0:
                # 旋不旋都可以，但是旋增加多样性，扩充数据集
                # 所以，旋
                image = rotate_image(image, angle_degree)
                return image, json_data

            px = px_ori.copy()
            py = py_ori.copy()
            if len(px.shape) < 2:
                px = np.expand_dims(px, axis=0)
                py = np.expand_dims(py, axis=0)
            #################################################################
            # 中间点，减去全图中心点
            for i in range(px.shape[0]): # 按停车位的个数来
                for j in range(px.shape[1]): # # 4个点 A B C D
                    # 奇偶正确处理
                    px[i,j] -= w // 2 + 0.5
                    py[i,j] -= h // 2 + 0.5
            """Rotate centralized marks with given angle in degree."""
            rx = px.copy()
            ry = py.copy()
            for i in range(px.shape[0]): # 按停车位的个数来
                # 按线逐点旋转，OA-OA' OB-OB' OC-OC' OD-OD'
                # 那就是中心旋转，O为图像中心
                # 当心超出图像边界啊（超出部分 能训）
                # 毕竟只要保证 目标点 A点不超出（和B点？）
                # 这个就存在了，所以trian阶段，必然会loss出现系统误差，
                # 不能绝对值回归大于1的值；但是测试阶段效果会很好
                # 黑色部分不影响。
                # label里真实存在！ 旋转就是太6了
                for j in range(px.shape[1]):
                    rx[i][j], ry[i][j] = rotate_vector([px[i][j], py[i][j]], angle_degree)
                    rx[i][j] += w // 2 + 0.5
                    ry[i][j] += h // 2 + 0.5

            json_data["points_x"] = rx
            json_data["points_y"] = ry

            # if overlap_check(rotated_marks, params):
            #     # overlap了；为什么要这样做？overlap不一样处理吗？
            #     json_data["points_x"] = px_ori # 返回原始值
            #     json_data["points_y"] = py_ori
            # else: # 没有overlap
            #     image = rotate_image(image, angle_degree)
            #     return image, json_data
            # # 只有原来的方法必然存在overlap！间隔车位！
            # # 注释掉
            image = rotate_image(image, angle_degree) # bingo！

            return image, json_data

        def flip_data(params, image, json_data, flip_type="vertical",
                      change_filp_type=0, p=1):
            assert flip_type in ["horizontal", "vertical"]
            if random.random() < change_filp_type:
                flip_type = 'horizontal' if flip_type == 'vertical' \
                    else 'vertical'
            if random.random() <= (1 - p): # x in the interval [0, 1)
                return image, json_data
            H, W, C = image.shape
            if random.random() <= params.h_and_v:
                image = np.flip(image, axis=0)
                json_data["points_y"] = [[H - k for k in p] for p in json_data["points_y"]]
                image = np.flip(image, axis=1)
                json_data["points_x"] = [[W - k for k in p] for p in json_data["points_x"]]
            else:
                if flip_type == "vertical":
                    image = np.flip(image, axis=0)
                    json_data["points_y"] = [[H - k for k in p] for p in json_data["points_y"]]
                else:  # horizontal
                    image = np.flip(image, axis=1)
                    json_data["points_x"] = [[W - k for k in p] for p in json_data["points_x"]]

            return image, json_data

        def new_flip_h_data(params, image, json_data):
            ''' h仍然flip，但标点交换，保持ABCD逆时针顺序 '''
            try:
                flip_r = params.new_flip_h_ratio
            except:
                flip_r = 0
            if random.random() < 1 - flip_r:
                return image, json_data
            H, W, C = image.shape
            image = np.flip(image, axis=0)
            # flip H
            json_data["points_y"] = [[H - k for k in p] for p in json_data["points_y"]]
            # swap AB & AD              B     A     D     C
            json_data['points_y'] = [[p[1], p[0], p[3], p[2]] for p in json_data['points_y']]
            json_data['points_x'] = [[p[1], p[0], p[3], p[2]] for p in json_data['points_x']]
            # after swap         new    A     B     C     D  新的标点
            return image, json_data

        def shift_w_data_pre(params, image, json_data):
            ''' exp91以前采用 '''
            # 对右子图（切图后），只shift right，W上只往右边shfit
            if random.random() < 1 - params.shift_w_ratio:
                return image, json_data
            H, W = params.input_size # 用int，不用round
            w_pixel = max(int(W * params.shift_w_pixel), 2) # 最少shift 1个pixel
            w_pixel = random.randrange(1, w_pixel, step=1)  # 左闭右开
            json_data["points_x"] = [[k + w_pixel for k in p] for p in json_data["points_x"]]
            matShift = np.float32([[1, 0, w_pixel], [0, 1, 0]]) # x + w_pixel, y + 0
            image = cv2.warpAffine(image, matShift, (W, H))
            return image, json_data

        def shift_w_data(params, image, json_data):
            # 对右子图（切图后），只shift right，W上只往右边shfit
            if random.random() < 1 - params.shift_w_ratio:
                return image, json_data
            H, W = params.input_size # 用int，不用round
            w_pixel = max(int(W * params.shift_w_pixel), 2) # 最少shift 1个pixel
            w_pixel = random.randrange(1, w_pixel, step=1)  # 左闭右开
            if 'shift_w_left':
                # 会受rotate影响，因为如下是所有点进行判断，故只有epoch训练得越多越好
                cnt = 0
                for p in json_data['points_x']:
                    if p[0] > 3/5 and p[1] > 3/5: # 3/5是个超参数
                       cnt += 1
                if cnt > 0 and cnt == len(json_data['points_x']):
                    # 1/2, 3/5, 4/5 侵略性不同
                    # 保留正样本，超出边界的留下来
                    # 用成2/3，因为此函数入口处，已经是0.75, 2/3*2/3=4/9,接近一半
                    # default: shift_w_left_ratio = 2/3
                    point_A = min([p[0] for p in json_data['points_x']])
                    point_B = min([p[1] for p in json_data['points_x']])
                    if random.random() < params.shift_w_left_ratio \
                        or w_pixel > W - point_A \
                        or w_pixel > W - point_B:
                            w_pixel *= -1 # 为负
            json_data["points_x"] = [[k + w_pixel for k in p] for p in json_data["points_x"]]
            matShift = np.float32([[1, 0, w_pixel], [0, 1, 0]]) # x + w_pixel, y + 0
            image = cv2.warpAffine(image, matShift, (W, H))
            return image, json_data

        def shift_w_data_v3(params, image, json_data):
            # 对右子图（切图后），左右shift
            if random.random() < 1 - params.shift_w_ratio:
                return image, json_data
            H, W = params.input_size
            w_pixel = max(int(W * params.shift_w_pixel), 2)  # 最少shift 1个pixel
            w_pixel = random.randrange(1, w_pixel, step=1)   # 左闭右开
            w_pixel = w_pixel if random.random() < 0.5 else -w_pixel
            # 先 shift
            json_data["points_x"] = [[k + w_pixel for k in p] for p in json_data["points_x"]]
            k = 0
            px = json_data["points_x"]
            py = json_data["points_y"]
            so = json_data["isOccupied"]
            npx, npy, nso = [], [], []
            for i in range(len(px)):
                # 后 过滤
                if px[i][0] + w_pixel > 1 - 123/128. or px[i][0] + w_pixel < 5/128.:
                    continue
                if px[i][1] + w_pixel > 1 - 123/128. or px[i][1] + w_pixel < 5/128.:
                    continue
                npx.append(px[i])
                npy.append(py[i])
                nso.append(so[i])
            json_data["points_x"] = npx
            json_data["points_y"] = npy
            json_data["isOccupied"] = nso
            matShift = np.float32([[1, 0, w_pixel], [0, 1, 0]])  # x + w_pixel, y + 0
            image = cv2.warpAffine(image, matShift, (W, H))
            return image, json_data

        def shift_h_data(params, image, json_data):
            if random.random() < 1 - params.shift_h_ratio:
                return image, json_data
            H, W = params.input_size
            h_pixel = max(int(H * params.shift_h_pixel), 1)
            h_pixel = random.randrange(-h_pixel, h_pixel, step=1)  # 左闭右开，H-1不考虑
            if h_pixel == 0: # 最少shift 1个pixel
                h_pixel = 1 if random.random() >= 0.5 else -1
            json_data["points_y"] = [[k + h_pixel for k in p] for p in json_data["points_y"]]
            matShift = np.float32([[1, 0, 0], [0, 1, h_pixel]]) # x + 0, y + h_pixel
            image = cv2.warpAffine(image, matShift, (W, H))
            return image, json_data

        # 归一化和提取数据，整理成delta_line的形式
        # 随机种子的产生算法与系统有关。Windows和Linux系统中产生的随机种子不同 (   )?
        def extract_label(json_data, params):
            marking_point = []
            for k in range(len(json_data['points_x'])):
                H, W = params.input_size
                px = [x for x in json_data['points_x'][k]]
                py = [y for y in json_data['points_y'][k]]
                # 是否过滤旋转到外部的点
                '''
                      ————————
                     |  A。   |   A点是目标角点
                     |      D.|
                  B. |        |   B点跑到外头及了，但是不影响! 只要A点在即可，半个车位也要显示出来。  
                     |C.      |
                     |        |
                     |        |
                     |        |
                     |________| img
                      
                '''
                # # after shift and rotate, clip
                if params.filter_outer_point:
                    # 也可考虑，超过5个像素内，给与通过，后续再做实验吧
                    # TODO (   )
                    ele = px[0]
                    if ele < 0 or ele > W - 1:
                        continue # 此时不能归一化做
                    ele = py[0]
                    if ele < 0 or ele > H - 1:
                        continue
                    # A点肯定是要保证的!
                    # B点可选
                    if params.filter_outer_point_B:
                        # 也可考虑，超过5个像素内，给与通过，后续再做实验吧
                        # TODO (   )
                        ele = px[1]
                        if ele < 0 or ele > W - 1:
                            continue # 此时不能归一化做
                        ele = py[1]
                        if ele < 0 or ele > H - 1:
                            continue
                    if params.filter_strict_point:
                        pix_x = 10
                        pix_y = 16
                        dsn = json_data['datasetName']
                        if 'in99' in dsn or 'benchmark' in dsn:
                            pix_y += int((1050-880)/2./(1050/384.)) # 31
                        if px[1] > W - pix_x or px[1] < pix_x or py[1] > H - pix_y or py[1] < pix_y:
                            continue
                        if px[0] > W - pix_x or px[0] < pix_x or py[0] > H - pix_y or py[0] < pix_y:
                            continue
                # 有误！这样把4个点都考虑了，和本意只考虑角点实现不一样，重新写
                # if params.filter_outer_point:
                #     # 也可考虑，超过5个像素内，给与通过，后续再做实验吧
                #     # TODO (   )
                #     for ele in px:
                #         if ele < 0 or ele > W - 1:
                #             continue # 此时不能归一化做
                #     for ele in py:
                #         if ele < 0 or ele > H - 1:
                #             continue
                px = [x / W for x in px]
                py = [y / H for y in py]
                if params.occupied_mode:
                    # For occupied
                    occ_value = json_data['isOccupied'][k]
                else:
                    # For available
                    occ_value = 1 - json_data['isOccupied'][k]
                infor = [
                    px[0], # x 目标点
                    py[0], # y 目标点
                    px[3] - px[0], # lenSepLine_x
                    py[3] - py[0], # lenSepLine_y
                    px[1] - px[0], # lenEntryLine_x
                    py[1] - py[0], # lenEntryLine_y
                    occ_value
                ]
                marking_point.append(MarkingPoint(*infor))
            return marking_point

        H, W, C = image.shape
        assert H / W == self.params.input_size[0] / self.params.input_size[1]

        # 这里先resize 和 后面再resize   有什么区别？
        # 感觉会影响到对齐，(先归一化更好) 嗯 这里归一化（尺度归一化），
        # 再做数据处理，然后归一化到0~1之间，竟然是0~1！而不是-1~1； 全0，黑色区域给干没了！真好!
        if H != self.params.input_size[0] and W != self.params.input_size[1]:
            image = cv2.resize(image, self.params.input_size[::-1], cv2.INTER_AREA)
            inp_H, inp_W = self.params.input_size
            for i in range(len(json_data['points_x'])):
                json_data['points_x'][i] = [k / W * inp_W for k in json_data['points_x'][i]]
                json_data['points_y'][i] = [k / H * inp_H for k in json_data['points_y'][i]]

        if self.mode != "train":
            marking_point = extract_label(json_data, self.params)
            image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=2) \
                if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.is_deploy:
                img = torch.tensor(image)
                img = img.permute(2,0,1).contiguous()
                return img, marking_point
            else:
                return ToTensor()(image), marking_point

        augment_hsv(image, self.params.hsv_p)
        image, json_data = flip_data(self.params, image,
                                     json_data,
                                     flip_type=self.params.filp_type,
                                     change_filp_type=self.params.change_filp_type,
                                     p=self.params.flip_p)
        # flip_h 后，交换AB点，实现为：交换AD、BC两条线；
        # 目的是为了保证rotate逆时针顺序的情况下，仍然能启用flip
        image, json_data = new_flip_h_data(self.params, image, json_data)

        # shift -> rotate ≠ rotate -> shift
        # 图像中心没变，以图像中心为坐标系原点进行旋转
        if random.random() < self.params.shift_before_rotate_ratio:
            image, json_data = shift_w_data(self.params, image, json_data)
            image, json_data = shift_h_data(self.params, image, json_data)
            image, json_data = rotate_data(image, json_data, self.params)
        else: # W H shift的顺序不影响，因为截断只发生在出口处的归一化前
            image, json_data = rotate_data(image, json_data, self.params)
            image, json_data = shift_w_data(self.params, image, json_data)
            image, json_data = shift_h_data(self.params, image, json_data)

        # 最后面来归一化
        marking_point = extract_label(json_data, self.params)

        image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), axis=2) \
            if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return ToTensor()(image), marking_point

    def __len__(self):
        # return size of dataset
        return len(self.sample_infor)


def collate_fn(batch):
    batch_out = {}
    for key in batch[0].keys():
        batch_out[key] = []

    for x in batch:
        for k, v in x.items():
            batch_out[k].append(v)

    for k, v in batch_out.items():
        if isinstance(v[0], torch.Tensor):
            batch_out[k] = torch.stack((v))

    return batch_out


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    _logger.info("Dataset type: {}, transform type: {}".format(
        params.dataset_type, params.transform_type))

    ds_type_list = []
    if params.dataset_type == "basic":
        train_ds = AutoParkingData(params, mode="train")
        eval_ds = AutoParkingData(params, mode="test")
        ds_type_list.extend(['train', 'test'])
    elif params.dataset_type == 'train':
        train_ds = AutoParkingData(params, mode="train")
        ds_type_list.append('train')
    elif params.dataset_type == 'test':
        eval_ds = AutoParkingData(params, mode="test")
        ds_type_list.append('test')
    else:
        raise ValueError("Unknown dataset_type in params, should in [basic, train, test]")

    dataloaders = {}
    if "train" in ds_type_list:
        # add train data loader
        train_dl = DataLoader(
            train_ds,
            batch_size=params.train_batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.cuda,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=3,
        )
        dataloaders["train"] = train_dl

    if "test" in ds_type_list:
        # chose val data loader for evaluate
        if params.eval_type in ["test", 'val']:
            dl = DataLoader(
                eval_ds,
                batch_size=params.eval_batch_size,
                shuffle=False,
                num_workers=params.num_workers_eval,
                pin_memory=params.cuda,
                collate_fn=collate_fn,
                prefetch_factor=3,
            )
        else:
            dl = None
            raise ValueError("Unknown eval_type in params, should in [val, test]")
        dataloaders[params.eval_type] = dl

    return dataloaders
