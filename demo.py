# -*- coding: utf-8 -*-
# !@time: 2020/10/10 下午9:55
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import torch

from config import RESTORE

retinanet = torch.load(RESTORE)
torch.save(retinanet.static_dict(), 'checkpoints/state_dict.pt')

