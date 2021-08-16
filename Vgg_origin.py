# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class VGG16_Origin(nn.Module):
    """
    Input is (N, 3, 448, 448)

    The basis is VGG-16
    The structure is as follows:
    features:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(64*224*224)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(128*112*112)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> pool3(256*56*56)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> pool4(512*28*28)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu -> pool5(512*14*14)
    classifier:
    ->  fc(4096)     -> relu -> dropout      -> fc(4096)             -> relu -> dropout
    fc:
    ->  fc(n_classes)
    """
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A VGG16 network is instantiated, pre-trained: {}, number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])   # 512*7*7,----> 4096
        self.fc = nn.Linear(in_features=4096, out_features=self._n_classes)

        if self._pretrained:
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

            # if use_two_step:
            #     # Freeze all layer in self.feature
            #     for params in self.features.parameters():
            #         params.requires_grad = False
            #     for params in self.classifier.parameters():
            #         params.requires_grad = False

    def forward(self, x):
        N = x.size(0)
        # assert x.size() == (N, 3, 448, 448)
        x = self.features(x)
        # assert x.size() == (N, 512, 14, 14)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(self.classifier(x))
        # assert x.size() == (N, self._n_classes)
        return x