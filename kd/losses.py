# -*- coding: utf-8 -*-
# !@time: 2020/12/8 下午7:45
# !@author: superMC @email: 18758266469@163.com
# !@fileName: losses.py
from abc import ABC

from torch import nn


class SpatialAttentionLoss(nn.Module, ABC):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, f1, f2):
        pass


class ChannelAttentionLoss(nn.Module, ABC):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, f1, f2):
        pass


class RelationFeatureLoss(nn.Module, ABC):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, f1, f2):
        pass
