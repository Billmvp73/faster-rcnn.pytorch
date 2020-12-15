# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class AlexNetFeature(nn.Module):
  def __init__(self, in_channels=3, feat_dim=9216):
    super(AlexNetFeature, self).__init__()
    self.feat_dim = feat_dim
    self.net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.AdaptiveAvgPool2d(output_size=(6, 6))
    )
  def forward(self, x):
    h = self.net(x)
    return h.view(-1, self.feat_dim)

class InstanceMask(nn.Module):
  def __init__(self, in_features=9216, out_features=56*56):
    super(InstanceMask, self).__init__()
    self.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=in_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features= 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=4096, out_features=out_features, bias=True),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 56, 56))
    )

  def forward(self, x):
    h = self.classifier(x)
    return h

class AlexNet(nn.Module):
  def __init__(self, in_channels=3, feat_dims=9216, out_features=56*56):
    super(AlexNet, self).__init__()
    self.features = AlexNetFeature(in_channels, feat_dims)
    self.instanceMask = InstanceMask(feat_dims, out_features)
  
  def forward(self, x):
    f = self.features(x)
    h = self.instanceMask(f)
    return h


class alexnet(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/unsup_alex.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    alex = AlexNet()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        alex.load_state_dict({k:v for k,v in state_dict.items() if k in alex.state_dict()})

    alex.classifier = nn.Sequential(*list(alex.instanceMask._modules.values())[:-2])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(alex.features._modules.values())[:-2])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = alex.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

