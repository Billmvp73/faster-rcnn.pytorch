
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
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable



class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

# unsup_video = nn.Sequential( # Sequential,
#0 	nn.Conv2d(3,96,(11, 11),(4, 4)),
#1 	nn.ReLU(),
#2	nn.MaxPool2d((3, 3),(2, 2)),
#3 	Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
#4 	nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
#5 	nn.ReLU(),
#6 	nn.MaxPool2d((3, 3),(2, 2)),
#7 	Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
#8 	nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
#9 	nn.ReLU(),
#10 	nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1)),
#11 	nn.ReLU(),
#12 	nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1)),
#13 	nn.ReLU(),
#14	nn.MaxPool2d((3, 3),(2, 2)),
#15 	nn.Sequential( # Sequential,
#16 		Lambda(lambda x: x.view(x.size(0),-1)), # View,
#17 		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(9216,4096)), # Linear,
#18 	),
#19	nn.ReLU(),
#20 	nn.Dropout(0.5),
#21 	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
#22 	nn.ReLU(),
#23 	nn.Dropout(0.5),
# )

class UnsupFeature(nn.Module):
  def __init__(self, in_channels=3):
    super(UnsupFeature, self).__init__()
    self.feature = nn.Sequential( # Sequential,
        nn.Conv2d(3,96,(11, 11),(4, 4)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
        #Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
        nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
        #Lambda(lambda x,lrn=lnn.SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
        nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),
        nn.MaxPool2d((3, 3),(2, 2)),
      )
    self.classifier = nn.Sequential(
        nn.Linear(9216,4096), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,4096), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
      )

  def forward(self, x):
    h = self.feature(x)
    x = self.classifier(h.view(-1, 4096))
    return x


class unsup_video(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = '/content/pretrained_model/unsup_video.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    unsup_video = UnsupFeature()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        if self.model_path.endswith('unsup_video.pth'):
          feature_dict ={}
          classifier_dict = {}
          for k, v in state_dict.items():
            num = int(k[:k.find('.')])
            if num >= 15:
              classifier_dict[str(int(k[:2]) -15)+ k[k.rfind('.'):]] = v
            elif num >= 4:
            #if k in unsup_video.features.state_dict():
              if num <= 7:
                feature_dict[str(num-1) + k[k.find('.'):]] = v
              else:
                feature_dict[str(num-2) + k[k.find('.'):]] = v
            else:
              feature_dict[k] = v
          unsup_video.feature.load_state_dict(feature_dict)
          unsup_video.classifier.load_state_dict(classifier_dict)
        else:
          unsup_video.load_state_dict({k:v for k,v in state_dict.items() if k in unsup_video.state_dict()})

    unsup_video.classifier = nn.Sequential(*list(unsup_video.classifier._modules.values()))

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(unsup_video.features._modules.values())[:-1])

    # Fix the layers before last conv:
    for layer in range(13):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = unsup_video.classifier

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