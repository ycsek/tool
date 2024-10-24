'''
Author: M·H·C Lab
Date: 24-10-2024 01:17:35
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 25-10-2024 01:33:37
'''
# TODO: Add the network structure here. It will include MLP, ConvNet, ResNet, LeNet, AlexNet and VGG.
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP
class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28*1 if channel == 1 else 32*32*3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
# ConvNet
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pool, img_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat 