import sys
import time
import thop
import torch
import torch.nn as nn

__all__ = ['lightnet']

NET_CONFIG = {  # k, inc, ouc, s, act, res
    "blocks2": [[3, 32, 64, 1, 0, 0]],  # 112
    "blocks3": [[3, 64, 128, 2, 1, 0], [3, 128, 128, 1, 0, 1]],     # 56
    "blocks4": [[3, 128, 256, 2, 1, 0], [3, 256, 256, 1, 0, 1]],    # 28
    "blocks5": [[3, 256, 512, 2, 1, 0], [5, 512, 512, 1, 1, 1],
                [5, 512, 512, 1, 1, 1], [5, 512, 512, 1, 1, 1],
                [5, 512, 512, 1, 1, 1], [5, 512, 512, 1, 0, 1]],    # 14
    "blocks6": [[5, 512, 1024, 2, 1, 0], [5, 1024, 1024, 1, 0, 0]], # 7
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBN(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(inc, ouc, kernel, stride,
                              padding, 1, groups, False)
        self.norm = nn.BatchNorm2d(ouc)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1,
                 need_act=1, residual=0):
        super().__init__()
        self.dwconv = ConvBN(inc, inc, kernel, stride, inc)
        self.pwconv = ConvBN(inc, ouc, 1)
        self.act = nn.ReLU(True)
        self.residual = residual
        self.need_act = need_act

    def forward(self, x):
        if self.residual == 1:
            shortcut = x
        x = self.dwconv(x)
        x = self.act(x)
        x = self.pwconv(x)
        if self.residual:
            x += shortcut
        if self.need_act == 1:
            x = self.act(x)
        return x


class LightNet(nn.Module):
    def __init__(self, r=1.0, with_act=True, in_chans=1,
                 need_residual=True, m=make_divisible):
        super().__init__()

        self.conv1 = ConvBN(in_chans, m(32 * r), 3, 2)

        for blk in range(2, 7):
            blocks = nn.Sequential(*[
                DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a,
                                   p if need_residual else 0)
                for i, (k, inc, ouc, s, a, p)
                in enumerate(NET_CONFIG[f"blocks{blk}"])
            ])
            setattr(self, f"blocks{blk}", blocks)

        self.act = nn.ReLU()
        self.with_act = with_act

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        outputs["stem"] = x
        for blk in range(2, 7):
            blocks = getattr(self, f"blocks{blk}")
            x = blocks(self.act(x))
            if blk-1 >= 3:
                outputs[f"dark{blk-1}"] = x
        if self.with_act:
            for k, v in outputs.items():
                outputs[k] = self.act(v)
        return outputs


def lightres(width, with_act=True, in_chans=1,
             need_residual=True, **kwargs):
    backbone = LightNet(width, with_act, in_chans,
                        need_residual, **kwargs)
    return backbone


# if 1:
if 0:
    import thop
    import time
    from utils import CSPDarknet as dark
    from van import *

    r = 0.125
    net0 = van(r, 1, in_chans=1)
    net1 = dark(1, 1, r, depthwise=True, act='relu')
    # net2 = lightres(r, in_chans=1, need_residual=True)
    net2 = lightres(r, in_chans=1, need_residual=False)
    nets = [net0, net1, net2]
    nets = [net2, net1, net0]
    nets = [net2]
    d = 1
    x = torch.randn(100, 1, 384, 128).cuda(d)
    # x = torch.randn(1, 1, 384, 128).cuda(d)

    for net in nets:
        dim = 1
        input_shape = (dim, 384, 128)
        input = torch.randn(1, *input_shape)
        flops, params = thop.profile(net, inputs=(input,), verbose=False)
        net.eval()
        output = net(input)['dark5']
        split_line = '=' * 30
        print(f'''
                {net.__class__}
                {split_line}
                Input  shape: {tuple(input.shape[1:])}
                Output shape: {tuple(output.shape[1:])}
                Flops: {flops / 10 ** 6:.3f} M
                Params: {params / 10 ** 3:.3f} K
                {split_line}''')
        if 1:
        # if 0:
            net.cuda(d)
            t1 = time.time()
            # for i in range(1000):
            #     s = net(x)
            for i in range(10):
                s = net(x)
            t2 = time.time()
            print(f'''
                    {round(t2 - t1, 4)} s
                    {int(torch.cuda.memory_allocated(d) / 1e6)} M
                    {int(torch.cuda.max_memory_allocated(d) / 1e6)} M
                    {int(torch.cuda.memory_reserved(d) / 1e6)} M
            ''')

        # print(net)
