import torch
import math
import pdb
from torch import nn
from model.utils import *
from model.van import van
from model.utils import CSPDarknet as dark
from model.utils import CSPDarknet
from model.lightnet import light
from model.lightres import lightres


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, image):
        # fpn output content features of [dark3, dark4, dark5]

        fpn_outs = self.backbone(image)

        outputs = self.head(fpn_outs)

        return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        params,
        in_dim,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        expand_ratio=0.5,
    ):
        super().__init__()
        model_type = params.model_type
        fix_expand = params.fix_expand
        self.is_deploy = params.is_BNC

        if model_type in ['dark', 'default']:
            self.backbone = dark(in_dim, depth, width,
                                 depthwise=depthwise, act=act)
			# self.backbone = CSPDarknet(in_dim, depth, width,
            #                      depthwise=depthwise, act=act)
        elif model_type == 'van':
            self.backbone = van(width, in_chans=in_dim)
        elif model_type == 'light':
            self.backbone = light(width, True, #params.lightnet_with_act,
                                  in_chans=in_dim)
        elif model_type == 'lightres':
            self.backbone = lightres(width, False, #params.lightnet_with_act,
                                  in_chans=in_dim)
        else:
            raise NotImplementedError

        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        if params.upsample_type == 'bilinear': # SNPE 不支持 nearest
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else: # default
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
            expansion=expand_ratio,
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
            expansion=1 if depthwise or fix_expand else expand_ratio,
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
            expansion=expand_ratio,
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
            expansion=expand_ratio,
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
        if self.is_deploy:
            input = input / 255.

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
        params,
        num_classes,
        head_width=1.0,
        yolo_width=1.0,
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
        self.params = params
        self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.Sequential(*[
            Conv(
                in_channels=int(256 * head_width),
                out_channels=int(256 * head_width),
                ksize=3,
                stride=1,
                act=act,
            ),
            Conv(
                in_channels=int(256 * head_width),
                out_channels=int(256 * head_width),
                ksize=3,
                stride=1,
                act=act,
            ),
        ])
        self.reg_convs = nn.Sequential(*[
            Conv(
                in_channels=int(256 * head_width),
                out_channels=int(256 * head_width),
                ksize=3,
                stride=1,
                act=act,
            ),
            Conv(
                in_channels=int(256 * head_width),
                out_channels=int(256 * head_width),
                ksize=3,
                stride=1,
                act=act,
            ),
        ])
        self.entryline_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sepline_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reg_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.obj_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.occ_preds = nn.Conv2d(
            in_channels=int(256 * head_width),
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.stems = BaseConv(
            in_channels=int(self.in_channels[-1] * yolo_width),
            out_channels=int(256 * head_width),
            ksize=1,
            stride=1,
            act=act,
        )

    def initialize_biases(self, prior_prob):

        b = self.reg_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.reg_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.entryline_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.entryline_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.sepline_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.sepline_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.obj_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

        b = self.occ_preds.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.occ_preds.bias = torch.nn.Parameter(b.view(-1),
                                                 requires_grad=True)

    def forward(self, x):

        x = self.stems(x)
        cls_x = x
        reg_x = x

        cls_feat = self.cls_convs(cls_x)
        entryline_output = self.entryline_preds(cls_feat)  # 2 channel
        sepline_output = self.sepline_preds(cls_feat)  # 2 channel

        reg_feat = self.reg_convs(reg_x)
        position_output = self.reg_preds(reg_feat)  # 2 channel
        confidence_output = self.obj_preds(reg_feat)# 1 channel
        occupied_output = self.occ_preds(reg_feat)  # 1 channel

        if self.params.without_exp:
            output = torch.cat([
                confidence_output,  # (0, 1)  confidence
                position_output,    # (0, 1)  offset x, y
                sepline_output,     # (-1, 1) sepline_x和y
                entryline_output,   # (-1, 1) entryline_x和y
                occupied_output,    # (0, 1)  occupied
            ], 1)
        else:
            output = torch.cat([
                confidence_output.sigmoid(),# (0, 1)  confidence
                position_output.sigmoid(),  # (0, 1)  offset x, y
                sepline_output.tanh(),      # (-1, 1) sepline_x和y
                entryline_output.tanh(),    # (-1, 1) entryline_x和y
                occupied_output.sigmoid(),  # (0, 1)  occupied
            ], 1)

        if self.params.is_BNC:
            b, c, h, w = output.shape
            output = output.reshape(b, c, -1).permute(0, 2, 1)

        return output


def get_model(params):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    backbone = YOLOPAFPN(params,
                         params.in_dim,
                         params.yolo_depth,
                         params.yolo_width,
                         in_channels=params.in_channels,
                         act=params.yolo_act,
                         depthwise=params.dwconv,
                         expand_ratio=params.expand_ratio)
    head = YOLOXHead(params,
                     params.yolox_params['num_classes'],
                     in_channels=params.in_channels,
                     head_width=params.head_width,
                     yolo_width=params.yolo_width,
                     act=params.yolo_act,
                     depthwise=params.dwconv)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model
