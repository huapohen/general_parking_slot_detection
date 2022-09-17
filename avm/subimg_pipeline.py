import os
import cv2
import sys
import json
import time
import shutil
import datetime
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from subprocess import check_call



class AVM:
    def __init__(self, args):
        super().__init__()
        self.mode = 'subimg'
        self.video_type = args.video_type
        self.video_dir = rf'{args.avm_dir}\{args.date}\{args.video_id}'
        self.src_video_path = self.video_dir + \
                              rf'\{args.video_id}.{self.video_type}'
        extract_video_frames = args.new_folder
        generate_input_imgs = args.new_folder
        self.src_frames_dir = self.video_dir + r'\src_frames'
        self.dst_frames_dir = self.video_dir + r'\dst_frames'
        self.avm_frames_dir = self.video_dir + r'\avm\test'
        self.avm_images_dir = self.video_dir + r'\avm\test\image'
        self.src_frames_path = self.src_frames_dir + rf'\%05d.jpg'

        # 创建文件夹
        if args.new_folder:
            if os.path.exists(self.avm_frames_dir):
                shutil.rmtree(self.avm_frames_dir)
        if args.new_folder:
            for path in [self.src_frames_dir, self.dst_frames_dir,
                         self.avm_frames_dir, self.avm_images_dir]:
                os.makedirs(path, exist_ok=True)

        # ffprobe获取视频帧信息
        cap = cv2.VideoCapture(self.src_video_path)
        self.H_f = H_f = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W_f = W_f = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 获取帧对齐384:128(3:1)比例有关信息
        self.aligned_info = self._align_resolution_info(H_f, W_f, self.mode)
        # self._prt_resolution_info()

        # ffmpeg提取帧
        if extract_video_frames:
            # self._extract_video_frames()
            self._extract_video_frames_cv2()

        # 生成数据集需要的输入帧
        if generate_input_imgs:
            self._generate_img_list_txt()
            # self._generate_aligned_imgs_avm()
            self._generate_aligned_imgs_subimg()


    def _align_resolution_info(self, H_frame, W_frame, mode='subimg'):
        '''
        有两个输入，供外部调用用
        '''
        ratio = 2 if mode == 'avm' else 1
        if W_frame % 2 == 1:
            tmp = int((W_frame + 1) / ratio * 3)
        else:
            tmp = int(W_frame / ratio * 3)
        if tmp >= H_frame:
            pad_H = tmp - H_frame
            pad_W = 1 if W_frame % 2 else 0
        else:
            pad_H = 3 * ratio - H_frame % (3 * ratio) # %3 & %2
            pad_W = int((H_frame + pad_H - tmp) / 3 * ratio)
        pad_left = int(pad_W / ratio)
        pad_right = pad_left if mode == 'avm' else 0
        if pad_W % 2:
            pad_left += 1
            pad_W += 1
        # 补顶端因为让顶端车位靠近图中央，便于conv卷
        # 补得太多了的话，up&bottom对半补，统一对半补
        # 全图3:2，2:3，1.5倍，可以
        pad_bottom = int(pad_H / 2)
        pad_up = pad_H - pad_bottom
        H_align = H_frame + pad_H
        W_align = W_frame + pad_W
        W_subimg = int(W_align / ratio)
        # 以列表的形式返回(需要返回的参数太多了)
        aligned_info = [H_frame, W_frame, W_subimg,
                        pad_H, pad_W, H_align, W_align,
                        pad_left, pad_right, pad_up, pad_bottom]
        return aligned_info

    def _get_video_info_ffprobe(self):
        # 删除之前的保存信息
        video_info_path = self.video_dir + r'\video_info.json'
        if os.path.exists(video_info_path):
            os.remove(video_info_path)
        # 获取视频帧信息
        check_call(['ffprobe',
                    '-i', self.src_video_path,
                    '-v', 'quiet',
                    '-show_streams',
                    '-print_format', 'json',
                    '>', video_info_path],
                   shell=True)
        # 打开保存的信息，以json的形式返回
        with open(video_info_path, 'r', encoding='utf-8') as f:
            video_info = json.load(f)['streams'][0]
        return video_info

    def _extract_video_frames(self):
        cnt_frames = int(self.video_info['nb_frames'])
        check_call(['ffmpeg',
                    '-i', self.src_video_path,
                    # '-v', 'quiet',
                    '-r', str(self.frame_rate),
                    '-f', 'image2',
                    self.src_frames_path],
                   shell=True)
        assert cnt_frames == len(os.listdir(self.src_frames_dir))

    def _extract_video_frames_cv2(self):
        cap = cv2.VideoCapture(self.src_video_path)
        frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # for k in tqdm(range(frames_len)):
        for k in range(frames_len):
            _, img = cap.read()
            save_path = self.src_frames_dir + rf'\{k+1:05d}.jpg'
            cv2.imwrite(save_path, img)
        cap.release()

    def _prt_resolution_info(self):
        H_frame, W_frame, W_subimg, \
        pad_H, pad_W, H_align, W_align, \
        pad_left, pad_right, pad_up, pad_bottom = self.aligned_info
        print(f'mode: {self.mode}')
        print(f'(H_frame, W_frame) = ({H_frame}, {W_frame})')
        print(f'(H_align, W_align) = ({H_align}, {W_align})' + \
        f'    {H_align} / ({W_subimg} + {W_subimg})' + \
        f'    {H_align} / {W_subimg} == 3')
        print(f'(H_subimg, W_subimg) = ({H_align}, {W_subimg})')
        print(f'pad_left={pad_left}, pad_right={pad_right}')
        print(f'pad_up={pad_up}, pad_bottom={pad_bottom}')

    def _generate_img_list_txt(self):
        img_list_path = self.avm_frames_dir + r'\img_list.txt'
        if os.path.exists(img_list_path):
            os.remove(img_list_path)
        f = open(img_list_path, 'a')
        len_files = len(os.listdir(self.src_frames_dir))
        for i, file in enumerate(os.listdir(self.src_frames_dir)):
            if self.mode == 'avm':
                name_left = 'l_' + file.replace('.jpg', '') + '\n'
                f.write(name_left)
                name_right = 'r_' + file.replace('.jpg', '')
                name_right += '\n' if i != len_files - 1 else ''
                f.write(name_right)
            else:
                name = file.replace('.jpg', '')
                name += '\n' if i != len_files - 1 else ''
                f.write(name)
        f.close()

    def _align_img_avm(self, image, aligned_info):
        '''
        有两个输入，供外部调用用
        '''
        # 获取384:128=3:1比例对齐信息
        H_frame, W_frame, W_subimg, \
        pad_H, pad_W, H_align, W_align, \
        pad_left, pad_right, pad_up, pad_bottom = aligned_info
        # 开始对齐
        mid_pix = int(W_frame / 2)
        # 补黑色区域，因为在数据增强训练时，已经有黑色区域
        zero = np.zeros([H_frame, pad_W, 3], dtype=np.uint8)
        merge_imgs = [image[:, :mid_pix], zero, image[:, mid_pix:]]
        img = np.concatenate(merge_imgs, axis=1) # W
        zero_u = np.zeros([pad_up, W_align, 3], dtype=np.uint8)
        zero_b = np.zeros([pad_bottom, W_align, 3], dtype=np.uint8)
        img = np.concatenate([zero_u, img, zero_b], axis=0) # H
        img_left = img[:, :int(W_align / 2)]
        img_right = img[:, int(W_align / 2):]
        # flip img_left or rotate 180 degree img_left
        # choose rotate, becase 车来回开的时候，都在右侧 (不是倒车)
        img_left = img_left[::-1, ::-1]
        # 返回比例对齐后的左右子图，（上下子图待定）
        return img_left, img_right

    def _align_img_subimg(self, image, aligned_info):
        '''
        有两个输入，供外部调用用
        '''
        # 获取384:128=3:1比例对齐信息
        H_frame, W_frame, W_subimg, \
        pad_H, pad_W, H_align, W_align, \
        pad_left, pad_right, pad_up, pad_bottom = aligned_info
        # 补黑色区域，因为在数据增强训练时，已经有黑色区域
        zero = np.zeros([H_frame, pad_W, 3], dtype=np.uint8)
        merge_imgs = [zero, image]
        img = np.concatenate(merge_imgs, axis=1) # W
        zero_u = np.zeros([pad_up, W_align, 3], dtype=np.uint8)
        zero_b = np.zeros([pad_bottom, W_align, 3], dtype=np.uint8)
        img = np.concatenate([zero_u, img, zero_b], axis=0) # H
        return img

    def _generate_aligned_imgs_avm(self):
        # 对齐图片
        # for file in tqdm(os.listdir(src_frames_dir)):
        for file in os.listdir(self.src_frames_dir):
            # 读取本地图片
            image = cv2.imread(self.src_frames_dir + rf'\{file}')
            # 对齐比例，获得左右子图
            img_left, img_right = self._align_img_avm(image, self.aligned_info)
            # 命名左右子图
            name_left = 'l_' + file
            path_left = self.avm_images_dir + rf'\{name_left}'
            name_right = 'r_' + file
            path_right = self.avm_images_dir + rf'\{name_right}'
            # 保存比例对齐后的图片
            cv2.imwrite(path_left, img_left)
            cv2.imwrite(path_right, img_right)
        print('finished _generate_aligned_imgs_avm')
        return

    def _generate_aligned_imgs_subimg(self):
        for file in os.listdir(self.src_frames_dir):
            image = cv2.imread(self.src_frames_dir + rf'\{file}')
            img = self._align_img_subimg(image, self.aligned_info)
            path = self.avm_images_dir + rf'\{file}'
            cv2.imwrite(path, img)
        return

    def restore_avm_from_imgs(self, x):
        ''' 外部函数 '''
        return

    def read_from_video(self):
        ''' 外部函数 '''
        return cv2.VideoCapture(self.src_video_path)

    def gengerate_single_input_avm(self, params, image):
        """ 外部函数 """
        # 获取比例对齐信息
        H_f, W_f, _ = image.shape
        aligned_info = self._align_resolution_info(H_f, W_f)
        # 比例对齐
        img_left, img_right = self._align_img_avm(image, aligned_info)
        # resize到model输入分辨率
        res = [aligned_info]
        H, W = params.input_size
        for img in [img_left, img_right]:
            h_, w_, _ = img.shape
            if h_ != H and w_ != W:
                img = cv2.resize(img, (W, H), cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # 灰度化
            # if params.in_dim == 1:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     img = np.expand_dims(img, axis=2)
            # # torch tensor
            # img = torchvision.transforms.ToTensor()(img)
            # img = torch.stack([img]) # add batch dimension
            res.append(img)
        return res

    def gengerate_single_input_subimg(self, params, image):
        """ 外部函数 """
        # 获取比例对齐信息
        H_f, W_f, _ = image.shape
        aligned_info = self._align_resolution_info(H_f, W_f)
        # 比例对齐
        img = self._align_img_subimg(image, aligned_info)
        # resize到model输入分辨率
        res = [aligned_info]
        H, W = params.input_size
        h_, w_, _ = img.shape
        if h_ != H and w_ != W:
            img = cv2.resize(img, (W, H), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res.append(img)
        return res

    def restore2fullimg(self, pred_left_right, params):
        return

    def plot_fullimg_slots(self, image, pred_fullimg, params):
        height = image.shape[0]
        width = image.shape[1]
        for confidence, marking_point in pred_fullimg:
            if confidence < params.confid_plot_inference:
                continue
            x, y, lenSepLine_x, lenSepLine_y, \
            lenEntryLine_x, lenEntryLine_y, isOccupied = marking_point
            p0_x = width * x
            p0_y = height * y
            p1_x = p0_x + width * lenEntryLine_x
            p1_y = p0_y + height * lenEntryLine_y
            if params.deltanorm_or_direction == 'deltanorm':
                p3_x = round(p0_x + width * lenSepLine_x)
                p3_y = round(p0_y + height * lenSepLine_y)
                p2_x = round(p1_x + width * lenSepLine_x)
                p2_y = round(p1_y + height * lenSepLine_y)
            elif params.deltanorm_or_direction == 'direction':
                H, W = params.input_size
                x_ratio, y_ratio = 1, H / W
                radian = math.atan2(marking_point.lenSepLine_y * y_ratio,
                                    marking_point.lenSepLine_x * x_ratio)
                sep_cos = math.cos(radian)
                sep_sin = math.sin(radian)
                p3_x = round(p0_x + 80 * sep_cos)
                p3_y = round(p0_y + 80 * sep_sin)
                p2_x = round(p1_x + 80 * sep_cos)
                p2_y = round(p1_y + 80 * sep_sin)
            else:
                raise NotImplementedError(f'Unkown method: '
                                          f'{params.deltanorm_or_direction}')
            p0_x, p0_y = round(p0_x), round(p0_y)
            p1_x, p1_y = round(p1_x), round(p1_y)
            # 画进入线目标点：逆时针旋转车位的进入线起始端点。
            # AB-BC-CD-DA 这里指A点
            cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), thickness=2)
            # 给目标点打上置信度，取值范围：0到1
            color = (255, 255, 255) if confidence > 0.7 else (100, 100, 255)
            if confidence < 0.3: color = (0, 0, 255)
            cv2.putText(image, f'{confidence:.3f}',  # 不要四舍五入
                        (p0_x + 6, p0_y - 4),
                        cv2.FONT_HERSHEY_PLAIN, 1, color)
            # 画上目标点坐标 (x, y)
            cv2.putText(image, f' ({p0_x},{p0_y})',
                        (p0_x, p0_y + 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # 在图像左上角给出图像的分辨率大小 (W, H)
            H, W = params.input_size
            cv2.putText(image, f'({W},{H})', (5, 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # 画车位掩码图，分三步
            # 第一步：画未被占概率，画是否可用提示
            avail = 'N' if isOccupied > 0.9 else 'Y'
            cv2.putText(image, avail,
                        ((p0_x + p1_x) // 2 + 10, (p0_y + p1_y) // 2 + 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            area = np.array([[p0_x, p0_y], [p1_x, p1_y],
                             [p2_x, p2_y], [p3_x, p3_y]])
            if avail == 'Y':
                color = (0, 255, 0)  # 置信度颜色
                if 1 - isOccupied < 0.5:
                    color = (0, 0, 255)  # 低置信度红色提示
                cv2.putText(image,
                            f'{1 - isOccupied:.3f}',  # 转换为"未被占用"置信度
                            ((p0_x + p2_x) // 2, (p0_y + p2_y) // 2 + 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, color)
            # 第二步：判断，如果可用，才画掩码图，否则全零
            mask = np.zeros_like(image)
            if 1 - isOccupied > 0.9:
                cv2.fillPoly(mask, [area], [0, 64, 0])
            # 第三步：把掩码图和原图进行叠加： output= a*i1+b*i2+c
            image = cv2.addWeighted(image, 1., mask, 0.5, 0)  # 数字是权重
            # 画车位进入线 AB
            cv2.arrowedLine(image, (p0_x, p0_y), (p1_x, p1_y), (0, 255, 0), 2, 8, 0, 0.2)
            # 画车位分割线 AD
            cv2.arrowedLine(image, (p0_x, p0_y), (p3_x, p3_y), (255, 0, 0), 2)  # cv2.line
            # 画车位分割线 BC
            if p1_x >= 0 and p1_x <= width - 1 and p1_y >= 0 and p1_y <= height - 1:
                cv2.arrowedLine(image, (p1_x, p1_y), (p2_x, p2_y), (33, 164, 255), 2)

        return image

    def load_checkpoints(self, params, model):
        model_path = os.path.join(params.model_dir, params.restore_file)
        map_location = None if params.cuda else torch.device('cpu')
        state = torch.load(model_path, map_location)
        try:
            model.load_state_dict(state["state_dict"])
        except:
            net_dict = model.state_dict()
            if "module" not in list(state["state_dict"].keys())[0]:
                state_dict = {
                    "module." + k: v
                    for k, v in state["state_dict"].items()
                    if "module." + k in net_dict.keys()
                }
            else:
                state_dict = {
                    k.replace("module.", ""): v
                    for k, v in state["state_dict"].items()
                    if k.replace("module.", "") in net_dict.keys()
                }
            net_dict.update(state_dict)
            model.load_state_dict(net_dict, strict=False)
        return model

