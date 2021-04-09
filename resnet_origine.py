# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class ResNet_Origine(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, net=18,n_classes=200, pretrained=True):
        super().__init__()
        print('| A ResNet{} network is instantiated, pre-trained: {}, number of classes: {}'.format(net,pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        if net == 18:
            resnet = torchvision.models.resnet18(pretrained=self._pretrained)
            in_features = 512
        elif net == 34:
            resnet = torchvision.models.resnet34(pretrained=self._pretrained)
            in_features = 512
        elif net == 50:
            resnet = torchvision.models.resnet50(pretrained=self._pretrained)
            in_features = 2048
        elif net == 101:
            resnet = torchvision.models.resnet101(pretrained=self._pretrained)
            in_features = 2048
        elif net == 152:
            resnet = torchvision.models.resnet152(pretrained=self._pretrained)
            in_features = 2048

        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=in_features, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

