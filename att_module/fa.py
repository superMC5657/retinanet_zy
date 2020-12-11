# -*- coding: utf-8 -*-
# !@time: 2020/12/11 下午3:36
# !@author: superMC @email: 18758266469@163.com
# !@fileName: fa.py
from abc import ABC

from torch import nn
import torch


class SSEModule(nn.Module, ABC):
    def __init__(self, in_planes, HxW, hidden_ratio=2):
        super().__init__()
        hidden_state = HxW // hidden_ratio
        self.heatmap = nn.Conv2d(in_planes, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(HxW, hidden_state, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state, HxW, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        y = self.heatmap(x).view(batch_size, -1)
        y = self.fc(y).view(batch_size, 1, width, height)
        return y


if __name__ == '__main__':
    x = torch.ones((1, 128, 10, 10))
    sa = SSEModule(128, 100)
    att = sa(x)
    print(att.size())
