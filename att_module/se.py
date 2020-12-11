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
        self.in_planes = in_planes
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


if __name__ == '__main__':
    cse = CSEModule(128)
    x = torch.ones((1, 128, 10, 10))
    ret = cse(x)
    print(ret.size())
