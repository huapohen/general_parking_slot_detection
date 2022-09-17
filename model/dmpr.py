"""Defines the detector network structure."""
import torch
from torch import nn
from model.utils import *


class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""
    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []

        # 0
        layers += conv_norm_act(in_dims=input_channel_size,
                                out_dims=depth_factor,
                                kernel_size=3,
                                stride=1)
        # 1
        layers += conv_norm_act(in_dims=depth_factor,
                                out_dims=2 * depth_factor,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        # 2
        depth_factor *= 2
        layers += detector_block(depth_factor)
        layers += conv_norm_act(in_dims=depth_factor,
                                out_dims=2 * depth_factor,
                                kernel_size=4,
                                stride=2,
                                padding=1)

        # 3
        depth_factor *= 2
        layers += detector_block(depth_factor)
        layers += conv_norm_act(in_dims=depth_factor,
                                out_dims=2 * depth_factor,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        # 4
        depth_factor *= 2
        layers += detector_block(depth_factor)
        layers += detector_block(depth_factor)
        layers += conv_norm_act(in_dims=depth_factor,
                                out_dims=2 * depth_factor,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        # 5
        depth_factor *= 2
        layers += detector_block(depth_factor)
        layers += detector_block(depth_factor)
        layers += conv_norm_act(in_dims=depth_factor,
                                out_dims=2 * depth_factor,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        depth_factor *= 2
        layers += detector_block(depth_factor)

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


class DMPR(nn.modules.Module):
    """Detector for point with direction."""
    def __init__(self, params):
        super(DMPR, self).__init__()
        self.extract_feature = YetAnotherDarknet(
            params.input_channel_size,
            params.depth_factor)  # no skip-connection here
        dim_inter = 32 * params.depth_factor
        layers = []
        layers += detector_block(dim_inter)
        layers += detector_block(dim_inter)
        layers += [
            nn.Conv2d(dim_inter,
                      params.output_channel_size,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        ]
        self.predict = nn.Sequential(*layers)

    def forward(self, image):

        tmp = self.extract_feature(image)
        prediction = self.predict(tmp)

        # point_pred contains 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        point_pred, angle_pred = torch.split(prediction, 4, dim=1)
        # prediction shape: bs, 6, 16, 16
        # point_pred shape: bs, 4, 16, 16
        # angle_pred shape: bs, 2, 16, 16
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        output = {}
        output['predictions'] = torch.cat((point_pred, angle_pred), dim=1)
        return output


def get_model(params):
    return (DMPR(params))
