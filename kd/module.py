# -*- coding: utf-8 -*-
# !@time: 2020/12/8 下午9:30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: module.py

from abc import ABC

from torch import nn


class SEModule(nn.Module, ABC):
    def __init__(self, channel_num, hidden_ratio=16):
        hidden_state = channel_num // hidden_ratio
        super().__init__()
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel_num, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, channel_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.global_average_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y.expand_as(x)
