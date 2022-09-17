"""Universal network struture unit definition."""
from torch import nn
import torch



def conv_norm_act(in_dims,
                  out_dims,
                  kernel_size,
                  stride,
                  padding='same',
                  norm=nn.BatchNorm2d,
                  act=nn.LeakyReLU(0.1)):
    """Define a convolution with norm and activation."""

    conv = nn.Conv2d(in_channels=in_dims,
                     out_channels=out_dims,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)
    norm = norm(out_dims)
    act = act
    layers = [conv, norm, act]
    return layers


def detector_block(in_dims):
    """Define a unit composite of a squeeze(1x1 conv) and expand(3x3 conv) unit."""
    layers = []
    layers += conv_norm_act(in_dims=in_dims,
                            out_dims=in_dims // 2,
                            kernel_size=1,
                            stride=1)
    layers += conv_norm_act(in_dims=in_dims // 2,
                            out_dims=in_dims,
                            kernel_size=3,
                            stride=1)
    return layers


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
            self,
            depth,
            in_channels=3,
            stem_out_channels=32,
            out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels,
                     stem_out_channels,
                     ksize=3,
                     stride=1,
                     act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2],
                                 in_channels * 2),
        )

    def make_group_layer(self,
                         in_channels: int,
                         num_blocks: int,
                         stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels,
                     in_channels * 2,
                     ksize=3,
                     stride=stride,
                     act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(*[
            BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
            BaseConv(
                filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            SPPBottleneck(
                in_channels=filters_list[1],
                out_channels=filters_list[0],
                activation="lrelu",
            ),
            BaseConv(
                filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
            BaseConv(
                filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
        ])
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(self,
                 in_dim,
                 dep_mul,
                 wid_mul,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False,
                 act="silu",
                 norm='bn'):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(in_dim, base_channels, ksize=3, act=act, norm=norm)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act, norm=norm),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     depth=base_depth,
                     depthwise=depthwise,
                     act=act,
                     norm=norm),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2,
                 base_channels * 4,
                 3,
                 2,
                 act=act,
                 norm=norm),
            CSPLayer(base_channels * 4,
                     base_channels * 4,
                     depth=base_depth * 3,
                     depthwise=depthwise,
                     act=act,
                     norm=norm),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4,
                 base_channels * 8,
                 3,
                 2,
                 act=act,
                 norm=norm),
            CSPLayer(base_channels * 8,
                     base_channels * 8,
                     depth=base_depth * 3,
                     depthwise=depthwise,
                     act=act,
                     norm=norm),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8,
                 base_channels * 16,
                 3,
                 2,
                 act=act,
                 norm=norm),
            SPPBottleneck(base_channels * 16,
                          base_channels * 16,
                          activation=act,
                          norm=norm),
            CSPLayer(base_channels * 16,
                     base_channels * 16,
                     depth=base_depth,
                     shortcut=False,
                     depthwise=depthwise,
                     act=act,
                     norm=norm),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "silu_":
        module = SiLU()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu",
                 norm='bn'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':  # psdet 用了in跑多卡训练
            self.norm = nn.InstanceNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 act="silu",
                 norm='bn'):
        super().__init__()
        self.dconv = BaseConv(in_channels,
                              in_channels,
                              ksize=ksize,
                              stride=stride,
                              groups=in_channels,
                              act=act,
                              norm=norm)
        self.pconv = BaseConv(in_channels,
                              out_channels,
                              ksize=1,
                              stride=1,
                              groups=1,
                              act=act,
                              norm=norm)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="silu",
                 norm='bn'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act,
                              norm=norm)
        self.conv2 = Conv(hidden_channels,
                          out_channels,
                          3,
                          stride=1,
                          act=act,
                          norm=norm)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels,
                               mid_channels,
                               ksize=1,
                               stride=1,
                               act="lrelu")
        self.layer2 = BaseConv(mid_channels,
                               in_channels,
                               ksize=3,
                               stride=1,
                               act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation="silu",
                 norm='bn'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=activation,
                              norm=norm)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels,
                              out_channels,
                              1,
                              stride=1,
                              act=activation,
                              norm=norm)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 act="silu",
                 norm='bn'):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act,
                              norm=norm)
        self.conv2 = BaseConv(in_channels,
                              hidden_channels,
                              1,
                              stride=1,
                              act=act,
                              norm=norm)
        self.conv3 = BaseConv(2 * hidden_channels,
                              out_channels,
                              1,
                              stride=1,
                              act=act,
                              norm=norm)
        module_list = [
            Bottleneck(hidden_channels,
                       hidden_channels,
                       shortcut,
                       1.0,
                       depthwise,
                       act=act,
                       norm=norm) for _ in range(depth)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act="silu",
                 norm='bn'):
        super().__init__()
        self.conv = BaseConv(in_channels * 4,
                             out_channels,
                             ksize,
                             stride,
                             act=act,
                             norm=norm)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


def get_grid(batch_size, h, w, start=0):
    # 这里x是图像的列
    if torch.cuda.is_available():
        xx = torch.arange(0, w).cuda()
        yy = torch.arange(0, h).cuda()
    else:
        xx = torch.arange(0, w)
        yy = torch.arange(0, h)
    xx = xx.view(1, -1).repeat(h, 1)
    yy = yy.view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(
        xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()

    grid[:, :
         2, :, :] = grid[:, :2, :, :] + start  # add the coordinate of left top
    return grid
