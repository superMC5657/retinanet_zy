# -*- coding: utf-8 -*-
# !@time: 2020/9/26 下午8:52
# !@author: superMC @email: 18758266469@163.com
# !@fileName: config.py
import torch

use_cuda = True
use_cuda = use_cuda and torch.cuda.is_available()

batch_size = 2
DISTRIBUTED = True
RESTORE = False
checkpoints_dir = 'checkpoints'
