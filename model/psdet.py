from dataclasses import replace
import torch
import math
import pdb
from torch import nn, relu
from model.utils import *
from einops import rearrange


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.backbone = YOLOPAFPN(params.yolo_depth,
                                  params.yolo_width,
                                  in_channels=params.in_channels,
                                  act=params.yolo_act)
        self.head = YOLOXHead(params.yolox_params['num_classes'],
                              params.yolo_width,
                              in_channels=params.in_channels,
                              act=params.yolo_act)
        self.detect_head = DetectHead(params.yolox_params['num_classes'],
                                      params.yolo_depth,
                                      params.yolo_width,
                                      in_channels=params.in_channels,
                                      act=params.yolo_act)

    def extract_position(self, output):
        batch_size, _, fea_h, fea_w = output.shape
        confidence = output[:, :1, :, :]
        position = output[:, 1:, :, :]  # related position

        grids = get_grid(batch_size, fea_h, fea_w)[:, :2, :, :]
        position = position + grids  # absolute position
        position[:, :1, :, :] /= fea_w  # normalize to 1
        position[:, 1:, :, :] /= fea_h

        position = rearrange(position,
                             'a b c d -> a (c d) b')  # h,w -> h*w for sort opr
        confidence = rearrange(confidence, 'a b c d -> a (c d) b').squeeze(-1)
        _, index = torch.sort(confidence)
        # bs, crop_patch_num, 每个图取置信度最高的crop_patch_num个坐标
        index_max = index[:, -self.params.crop_patch_num:]
        bs_index = torch.arange(batch_size,
                                device=confidence.device).unsqueeze(1)
        # bs, 2, crop_patch_num
        position_sel = position[bs_index, index_max, :].permute(0, 2, 1)
        position_sel = position_sel.detach()  # 需不需要detach呢？

        return position_sel

    def crop_patches(self, position_sel, image):
        batch_size = image.shape[0]
        position_sel[:, 0, :] *= image.shape[-1]  # width
        position_sel[:, 1, :] *= image.shape[-2]  # height
        position_sel = position_sel - self.params.crop_patch // 2  # 中心点-->左上角点
        # 把多个位置放到batch上做并行 TODO：判断是不是需要换一下x和y的位置
        # 同一个batch内的数据会挨着排到第一维上，于是可以直接和后边的image_repeat做运算
        position_sel_int = position_sel.clone().int()
        position_sel_int = rearrange(
            position_sel_int, 'a b c -> (a c) b').unsqueeze(-1).unsqueeze(-1)

        patch_indices = get_grid(batch_size * self.params.crop_patch_num,
                                 self.params.crop_patch,
                                 self.params.crop_patch,
                                 position_sel_int)[:, :2, :, :]  # 获取待裁patch坐标

        image_repeat = image.repeat_interleave(self.params.crop_patch_num,
                                               axis=0)  # 每张图要裁剪n次，为了并行复制一下

        num_batch_repeat, num_channels, height, width = image_repeat.shape
        out_height, out_width = patch_indices.shape[-2:]
        max_x, max_y = width - 1, height - 1

        x_grid_flat = patch_indices[:, 0, ...].reshape([-1])
        y_grid_flat = patch_indices[:, 1, ...].reshape([-1])
        x_grid_flat = torch.clamp(x_grid_flat, 0, max_x)  # same as np.clip
        y_grid_flat = torch.clamp(y_grid_flat, 0, max_y)

        # 一维坐标索引计算
        dim1, dim2 = width * height, width
        base = torch.arange(0, num_batch_repeat, device=image.device).int()
        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)
        y_grid_flat = base + y_grid_flat * dim2
        grid_flat = y_grid_flat + x_grid_flat

        # 并行取patch
        image_repeat = image_repeat.permute(0, 2, 3, 1)
        image_repeat = image_repeat.reshape([-1, num_channels]).float()

        grid_flat = grid_flat.unsqueeze(-1).long()
        grid_flat = grid_flat.expand(out_height * out_width * num_batch_repeat,
                                     num_channels)
        image_patches = torch.gather(image_repeat, 0, grid_flat)
        image_patches = image_patches.reshape(num_batch_repeat, out_height,
                                              out_width, num_channels)
        image_patches = image_patches.permute(
            0, 3, 1, 2)  # bs*crop_patch_num, c, crop_patch, crop_patch

        # 归一化回去，但是这里要注意是patch左上角处的坐标，方便后续embedding_results
        position_sel[:, 0, :] /= image.shape[-1]  # width
        position_sel[:, 1, :] /= image.shape[-2]  # height
        return image_patches, position_sel

    def embedding_results(self, output_2, position_sel, index_max):
        # output_2: bs*crop_patch_num, 6, 1, 1, channels: confidence, shape_type, position_x, position_y, sin, cos
        # position_sel: bs, 2, crop_patch_num
        # index_max: # bs, crop_patch_num

        fea_size = 16
        output_2 = output_2.squeeze(-1).squeeze(-1)
        output_2 = rearrange(output_2,
                             '(a b) c -> a c b',
                             b=self.params.crop_patch_num)
        batch_size = output_2.shape[0]

        # 根据裁patch的坐标恢复position与gt格式对应
        position_sel *= fea_size
        output_2[:, [2, 3], :] += position_sel
        grid_index = output_2[:, [2, 3], :].floor()
        grid_index = grid_index[:, 0, :] + grid_index[:, 1, :] * fea_size
        grid_index = grid_index.long()
        # 预测的x和y分别是矩阵的列与行
        output = torch.zeros((batch_size, 6, fea_size, fea_size),
                             device=output_2.device)  # 16 is the fea_size
        bs_index = torch.arange(batch_size,
                                device=output_2.device).unsqueeze(1)

        output = rearrange(output, 'a b c d -> a (c d) b')
        output_2 = rearrange(output_2, 'a b c -> a c b')
        output[bs_index, grid_index, :] = output_2
        output = rearrange(output, 'a (c d) b -> a b c d', c=fea_size)
        output[:, 2:4, :, :] -= output[:, 2:4, :, :].floor()

        return output

    def forward(self, image):

        # stage 1
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(image)
        output_1 = self.head(fpn_outs)

        # crop patch
        position_sel = self.extract_position(
            output_1)  # position_sel是归一化的且指向所选角点
        image_patches, position_sel = self.crop_patches(position_sel, image)

        # stage 2
        output_2 = self.detect_head(image_patches)

        output_2 = self.embedding_results(output_2, position_sel)

        outputs = {}
        outputs["output_1"] = output_1
        outputs["output_2"] = output_2
        return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width),
                                      int(in_channels[1] * width),
                                      1,
                                      1,
                                      act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width),
                                     int(in_channels[0] * width),
                                     1,
                                     1,
                                     act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width),
                             int(in_channels[0] * width),
                             3,
                             2,
                             act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width),
                             int(in_channels[1] * width),
                             3,
                             2,
                             act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = pan_out0
        return outputs


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.reg_convs = nn.Sequential(*[
            Conv(
                in_channels=int(256 * width),
                out_channels=int(256 * width),
                ksize=3,
                stride=1,
                act=act,
            ),
            Conv(
                in_channels=int(256 * width),
                out_channels=int(256 * width),
                ksize=3,
                stride=1,
                act=act,
            ),
        ])

        self.reg_preds = nn.Conv2d(
            in_channels=int(256 * width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_preds = nn.Conv2d(
            in_channels=int(256 * width),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.stems = BaseConv(
            in_channels=int(self.in_channels[-1] * width),
            out_channels=int(256 * width),
            ksize=1,
            stride=1,
            act=act,
        )

    def initialize_biases(self, prior_prob):

        b = self.obj_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

    def forward(self, x):

        x = self.stems(x)
        reg_x = x

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)
        confidence_output = self.obj_preds(reg_feat)

        output = torch.cat([
            confidence_output.sigmoid(),
            position_output.sigmoid(),
        ], 1)

        return output


class DetectHead(nn.Module):
    def __init__(
        self,
        num_classes,
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        base_channels = int(width * 32)  # 32
        base_depth = max(round(depth * 3), 1)  # 3

        Conv = DWConv if depthwise else BaseConv

        self.stems = Focus(3, base_channels, ksize=3, act=act)
        # self.stems = BaseConv(3, base_channels, ksize=3, stride=1, act=act)
        # self.stems = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,
        #         out_channels=base_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ), nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     depth=base_depth,
                     depthwise=depthwise,
                     act=act))

        self.cls_convs = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     depth=base_depth,
                     depthwise=depthwise,
                     act=act))

        self.reg_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

        self.obj_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

        self.cls_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

    def initialize_biases(self, prior_prob):

        b = self.obj_preds[0].bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_preds[0].bias = torch.nn.Parameter(b.view(-1),
                                                    requires_grad=True)

    def forward(self, x):

        x = self.stems(x)
        cls_x = x
        reg_x = x

        cls_feat = self.cls_convs(cls_x)
        angle_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)
        confidence_output = self.obj_preds(reg_feat)

        output = torch.cat([
            confidence_output.sigmoid(),
            position_output.sigmoid(),
            angle_output.tanh()
        ], 1)

        return output


class DetectHead(nn.Module):
    def __init__(
        self,
        num_classes,
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        base_channels = int(width * 32)  # 32
        base_depth = max(round(depth * 3), 1)  # 3

        Conv = DWConv if depthwise else BaseConv

        self.stems = Focus(3, base_channels, ksize=3, act=act)
        # self.stems = BaseConv(3, base_channels, ksize=3, stride=1, act=act)
        # self.stems = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,
        #         out_channels=base_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ), nn.ReLU(inplace=True))
        self.reg_convs = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     depth=base_depth,
                     depthwise=depthwise,
                     act=act))

        self.cls_convs = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     depth=base_depth,
                     depthwise=depthwise,
                     act=act))

        self.reg_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

        self.obj_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

        self.cls_preds = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.AdaptiveAvgPool2d(1))

    def initialize_biases(self, prior_prob):

        b = self.obj_preds[0].bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_preds[0].bias = torch.nn.Parameter(b.view(-1),
                                                    requires_grad=True)

    def forward(self, x):

        x = self.stems(x)
        cls_x = x
        reg_x = x

        cls_feat = self.cls_convs(cls_x)
        angle_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)
        confidence_output = self.obj_preds(reg_feat)

        output = torch.cat([
            confidence_output.sigmoid(),
            position_output.sigmoid(),
            angle_output.tanh()
        ], 1)

        return output


def get_model(params):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    model = YOLOX(params)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)  # 这个经验值改不改再说
    model.detect_head.initialize_biases(1e-2)
    return model