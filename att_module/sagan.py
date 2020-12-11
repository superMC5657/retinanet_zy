# -*- coding: utf-8 -*-
# !@time: 2020/12/10 上午8:48
# !@author: superMC @email: 18758266469@163.com
# !@fileName: sagan.py
from abc import ABC

from torch import nn
import torch


class Self_Attn(nn.Module, ABC):
    """ Self attention Layer"""

    def __init__(self, in_planes, hidden_ratio, alpha=1):
        super(Self_Attn, self).__init__()
        hidden_state = in_planes // hidden_ratio

        self.query_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_planes, out_channels=hidden_state, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1) * alpha)

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        batch_size, channel_num, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1).contiguous()  # B X CX(N)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # transpose check
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1).contiguous())
        out = out.view(batch_size, channel_num, width, height)
        out = self.gamma * out + x

        return out, attention


if __name__ == '__main__':
    x = torch.ones((1, 128, 10, 10))
    sa = Self_Attn(128, 8)
    ret, att = sa(x)
    print(ret.size(), att.size())
