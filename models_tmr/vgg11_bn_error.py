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


class VGG11(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
                 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                 ReLU(inplace=True),
                 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

