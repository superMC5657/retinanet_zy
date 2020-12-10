# -*- coding: utf-8 -*-
# !@time: 2020/12/10 上午12:04
# !@author: superMC @email: 18758266469@163.com
# !@fileName: da.py
# dual attention


from abc import ABC

import torch
from torch import nn


class ChannelAttention(nn.Module, ABC):
    def __init__(self, alpha):
        super(ChannelAttention, self).__init__()

    def forward(self, x):
        batch_size, channel_num, height, width = x.size()
        x1 = x.reshape(batch_size, channel_num, -1)
        x2 = x1.transpose(1, 2).contiguous()
        attn = torch.matmul(x1, x2)


class SpatialAttention(nn.Module, ABC):
    def __init__(self, in_planes, hidden_ratio, beta=0.1):
        super(SpatialAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        x1 = x.reshape(batch_size, channel_num, -1)
        x2 = x1.transpose(1, 2).contiguous()
        attn = torch.bmm(x2, x1)
        attn = self.softmax(attn)
        return attn


if __name__ == '__main__':
    sa = SpatialAttention(128, 16, beta=0.1)
    x = torch.ones((1, 128, 10, 10))
    ret = sa(x)
    print(ret.size())
