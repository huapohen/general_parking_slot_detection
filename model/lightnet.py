
import sys
import time
import thop
import torch
import torch.nn as nn


__all__ = ['lightnet']


NET_CONFIG = { # k, inc, ouc, s, act
    "blocks2": [[3,  32, 64, 1, 0]],                          # 112
    "blocks3": [[3,  64, 128, 2, 1], [3, 128, 128, 1, 0]],    # 56
    "blocks4": [[3, 128, 256, 2, 1], [3, 256, 256, 1, 0]],    # 28
    "blocks5": [[3, 256, 512, 2, 1], [5, 512, 512, 1, 1],
                [5, 512, 512, 1, 1], [5, 512, 512, 1, 1],
                [5, 512, 512, 1, 1], [5, 512, 512, 1, 0]],    # 14
    "blocks6": [[5, 512, 1024,2, 1], [5, 1024,1024,1, 0]],    # 7
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1,
                 groups=1, need_act=1, act=nn.ReLU):
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(inc, ouc, kernel, stride, padding, 1, groups, False)
        self.norm = nn.BatchNorm2d(ouc)
        self.need_act = need_act
        if need_act == 1:
            self.act = act(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.need_act == 1:
            x = self.act(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self, inc, ouc, kernel=3, stride=1, need_act=1):
        super().__init__()
        self.dwconv = ConvBNAct(inc, inc, kernel, stride, inc)
        self.pwconv = ConvBNAct(inc, ouc, 1, 1, need_act=1)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


class LightNet(nn.Module):
    def __init__(self, r=1.0, with_act=False, in_chans=1, m=make_divisible):
        super().__init__()

        self.conv1 = ConvBNAct(in_chans, m(32 * r), 3, 2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a)
            for i, (k, inc, ouc, s, a) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a)
            for i, (k, inc, ouc, s, a) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a)
            for i, (k, inc, ouc, s, a) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a)
            for i, (k, inc, ouc, s, a) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(m(inc * r), m(ouc * r), k, s, a)
            for i, (k, inc, ouc, s, a) in enumerate(NET_CONFIG["blocks6"])
        ])
        self.act = nn.ReLU()
        self.with_act = with_act

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        # outputs["stem"] = x     # 1/2
        x = self.blocks2(x)
        # outputs["dark1"] = x    # 1/2
        x = self.blocks3(self.act(x))
        # outputs["dark2"] = x    # 1/4
        x = self.blocks4(self.act(x))
        outputs["dark3"] = x    # 1/8
        x = self.blocks5(self.act(x))
        outputs["dark4"] = x    # 1/16
        x = self.blocks6(self.act(x))
        outputs["dark5"] = x    # 1/32

        if self.with_act:
            for k, v in outputs.items():
                outputs[k] = self.act(v)
        return outputs


def light(width, with_act=False, **kwargs):
    backbone = LightNet(r=width, with_act=False, **kwargs)
    return backbone





# if 1:
if 0:
    import thop
    import time
    from utils import CSPDarknet as dark
    from van import *
    r = 0.125
    net0 = van(r, 1, in_chans=1)
    net1 = dark(1, 1, r, depthwise=True, act='relu') # 0.0173 s
    net2 = light(r, in_chans=1) # 0.0049 s
    nets = [net0, net1, net2]
    # nets = [net2, net1, net0]
    d = 2
    x = torch.randn(1, 1, 384, 128).cuda(d)
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
            for i in range(1):
                s = net(x)
            t2 = time.time()
            print(f'''
                    {round(t2 - t1, 4)} s
                    {int(torch.cuda.memory_allocated(d) / 1e6)} M
                    {int(torch.cuda.max_memory_allocated(d) / 1e6)} M
                    {int(torch.cuda.memory_reserved(d) / 1e6)} M
            ''')
            del net
            torch._C._cuda_emptyCache()
        # print(net)


# 反复执行多次，取平均值 (   )
"""

                <class '__main__.LightNet'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 14.669 M
                Params: 65.720 K
                ==============================

                2.5008 s
                6 M
                12 M
                12 M
        

                <class 'utils.CSPDarknet'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 27.814 M
                Params: 156.528 K
                ==============================

                11.4257 s
                12 M
                23 M
                25 M
                
                
                <class 'van.VAN'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 181.008 M
                Params: 814.688 K
                ==============================

                9.1621 s
                34 M
                63 M
                67 M

"""


"""

                <class 'van.VAN'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 181.008 M
                Params: 814.688 K
                ==============================

                9.1927 s
                32 M
                62 M
                65 M


                <class 'utils.CSPDarknet'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 27.814 M
                Params: 156.528 K
                ==============================

                11.0497 s
                12 M
                22 M
                25 M
        

                <class '__main__.LightNet'>
                ==============================
                Input  shape: (1, 384, 128)
                Output shape: (128, 12, 4)
                Flops: 14.669 M
                Params: 65.720 K
                ==============================

                2.5134 s
                7 M
                22 M
                25 M
        
"""