# -*- coding: utf-8 -*-
# !@time: 2020/12/8 下午9:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: se.py

from abc import ABC

import torch
from torch import nn


class CSEModule(nn.Module, ABC):
    def __init__(self, in_planes, hidden_ratio=16):
        hidden_state = in_planes // hidden_ratio
        super().__init__()
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, height, width = x.size()
        y = self.global_average_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return y


class SSEModule(nn.Module, ABC):
    def __init__(self, HxW, hidden_ratio=10):
        super().__init__()
        hidden_state = HxW // hidden_ratio
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(HxW, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, HxW, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, height, width = x.size()
        y = x.view(batch_size, channel_num, -1)
        y = y.transpose(1, 2).contiguous()
        y = self.global_average_pool(y).view(batch_size, -1)
        y = self.fc(y).view(batch_size, 1, height, width)
        return y


if __name__ == '__main__':
    cse = CSEModule(128)
    sse = SSEModule(100)
    x = torch.ones((1, 128, 10, 10))
    ret = cse(x)
    ret2 = sse(x)
    print(ret.size())
    print(ret2.size())
