# -*- coding: utf-8 -*-
# !@time: 2020/10/10 下午9:55
# !@author: superMC @email: 18758266469@163.com
# !@fileName: demo.py
import torch

from retinanet.model import retinanet50
from torch.utils.tensorboard import SummaryWriter

model = retinanet50(num_classes=80, pretrained=False)
static_dict = torch.load('checkpoints/coco_resnet_50_map_0_335_state_dict.pt')
model.load_state_dict(static_dict)
input_tensor = torch.rand(size=(1, 3, 512, 512))
with SummaryWriter(comment="retinanet") as sw:
    sw.add_graph(model, (input_tensor,))
