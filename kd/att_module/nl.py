# -*- coding: utf-8 -*-
# !@time: 2020/12/10 上午12:05
# !@author: superMC @email: 18758266469@163.com
# !@fileName: nl.py
from abc import ABC

from torch import nn


class NonLocal(nn.Module, ABC):
    def __init__(self, in_planes, ):
        super(NonLocal, self).__init__()

    def forward(self, x):
        return x
