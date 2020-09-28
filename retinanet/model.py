# -*- coding: utf-8 -*-
# !@time: 2020/9/26 15 30
# !@author: superMC @email: 18758266469@163.com
# !@fileName: model.py
from abc import ABC

import torch
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, BottleNeck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)


class PyramidFeatures(nn.Module, ABC):
    '''
    FPN
    '''

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        #
        self.P5_1 = conv1x1(C5_size, feature_size)
        self.P5_Upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = conv3x3(feature_size, feature_size)

        self.P4_1 = conv1x1(C4_size, feature_size)
        self.P4_Upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = conv3x3(feature_size, feature_size)

        self.P3_1 = conv1x1(C3_size, feature_size)
        self.P3_2 = conv3x3(feature_size, feature_size)

        self.P6 = conv3x3(C5_size, feature_size, stride=2)

        self.relu = nn.ReLU()
        self.P7 = conv3x3(feature_size, feature_size, stride=2)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_Upsample_x = self.P5_Upsample(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_Upsample_x + P4_x
        P4_upsample_x = self.P4_Upsample(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P4_upsample_x + P3_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.relu(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module, ABC):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = conv3x3(num_features_in, feature_size)
        self.conv2 = conv3x3(feature_size, feature_size)
        self.conv3 = conv3x3(feature_size, feature_size)
        self.conv4 = conv3x3(feature_size, feature_size)

        self.output = conv3x3(feature_size, num_anchors * 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.output(out)

        out = out.premute(0, 2, 3, 1).contiguous()

        out = out.view(out.shape[0], -1, 4)
        return out


class ClassificationModel(nn.Module, ABC):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel).__init__()

        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv3x3(num_features_in, feature_size)
        self.conv2 = conv3x3(feature_size, feature_size)
        self.conv3 = conv3x3(feature_size, feature_size)
        self.conv4 = conv3x3(feature_size, feature_size)

        self.output = conv3x3(feature_size, num_anchors * num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.output(out)
        out = self.sigmoid(out)

        out = out.premute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, self.num_classes)
        return out


class ResNet(nn.Module, ABC):
    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layers(block, 64, layers[0])
        self.layer2 = self.__make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self.__make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self.__make_layers(block, 256, layers[3], stride=2)

    def __make_layers(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        如果batch_size太小就需要freeze_bn
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BottleNeck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BottleNeck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BottleNeck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
