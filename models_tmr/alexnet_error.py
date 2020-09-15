import torch
import torch.nn as nn
import torch.nn.functional as F

'''
from myfunction.Conv2D import Conv2d
from myfunction.Linear import Linear
from myfunction.container import Sequential, ReLU, MaxPool2d, Dropout
from myfunction.Qcode import float2Qcode
'''
from function_tmr.Conv2D import Conv2d
from function_tmr.Linear import Linear
from function_tmr.container import Sequential, ReLU, MaxPool2d, Dropout
from function_tmr.Qcode import float2Qcode

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 6 * 6, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

