# -*- coding: utf-8 -*-
# !@time: 2020/9/26 15 30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: losses.py
from abc import ABC

import numpy as np
import torch
from torch import nn
from torchvision.ops.boxes import box_iou
from config import use_cuda


class FocalLoss(nn.Module, ABC):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.targets_std = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
        if use_cuda:
            self.targets_std.cuda()

    @staticmethod
    def xyxy2ctr_xywh(boxes):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * w
        ctr_y = boxes[:, 1] + 0.5 * h
        return ctr_x, ctr_y, w, h

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25  # 平衡正负样本本身的不均衡
        gamma = 2.0  # 可以关注到困难的错分的样本
        batch_size = classifications.shape[0]
        classifications_losses = []
        regressions_losses = []
        anchor = anchors[0, :, :]  # 这是因为retinanet使用所有的anchor进行训练 所以anchors中的anchor相同

        anchor_ctr_x, anchor_ctr_y, anchor_widths, anchor_heights = self.xyxy2ctr_xywh(anchors)

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)  # log0 0不在定义域

            if bbox_annotation.shape[0] == 0:  # 这张图片没有object 只计算分类损失

                alpha_factor = torch.ones(classification.shape) * alpha

                if use_cuda:
                    alpha_factor = alpha_factor.cuda()

                # background focal_bce_loss
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                bce = -(torch.log(1.0 - classification))
                cls_loss = focal_weight * bce

                classifications_losses.append(cls_loss.sum())
                regressions_losses.append(torch.tensor(0).float())
                continue
            iou = box_iou(anchor, bbox_annotation[:, :4])
            iou_max, iou_argmax = torch.max(iou, dim=1)

            targets = torch.ones(classification.shape) * -1
            if use_cuda:
                targets = targets.cuda()
            targets[torch.lt(iou_max, 0.4), :] = 0
            positive_indices = torch.ge(iou_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[iou_argmax, :]
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * alpha
            if use_cuda:
                alpha_factor = alpha_factor.cuda()
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1 - alpha_factor)  # 有object用0.25 没有用0.75
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            if use_cuda:
                cls_loss.cuda()

            classifications_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]

                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = self.xyxy2ctr_xywh(assigned_annotations)

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()
                targets = targets / self.targets_std

                negative_indices = 1 + (~positive_indices)
                # negative_indices = torch.ge(0.5, iou_max) - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regressions_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0),
                                               0.5 * 9.0 * torch.pow(regression_diff, 2),
                                               regression_diff - 0.5 / 9.0)
                regressions_losses.append(regressions_loss.mean())
            else:
                regressions_losses.append(torch.tensor(0).float())

        return torch.stack(classifications_losses).mean(dim=0, keepdim=True), torch.stack(regressions_losses).mean(
            dim=0, keepdim=True)
