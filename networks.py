'''
Author: Jason Shi
Date: 24-10-2024 11:16:21
Last Editors: Jason
Contact Last Editors: D23090120503@cityu.edu.mo
LastEditTime: 25-10-2024 02:33:09
'''

# TODO: Add the network structure here. It will include MLP, ConvNet, ResNet, LeNet, AlexNet and VGG.
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP(多层感知器)


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

# ConvNet(卷积神经网络)


class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pool, img_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feature = self._make_layers(
            channel, net_width, net_depth, net_act, net_norm, net_pool, img_size)
        num_feat = shape_feature[0] * shape_feature[1] * shape_feature[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # Activation Function (激活函数)
    def _get_activation(self, net_act):
        if net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'tanh':
            return nn.Tanh()
        elif net_act == 'softmax':
            return nn.Softmax(dim=1)
        else:
            exit("Unknown Activation Function: %s" % net_act)

    # Pooling Layer (池化层)
    def _get_pool(self, net_pool):
        if net_pool == 'max':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pool == 'avg':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pool == 'global':
            return nn.AdaptiveAvgPool2d(1)
        elif net_pool == 'none':
            return None
        else:
            exit("Unknown net_pooling: %s" % net_pool)

    # Normalization Layer (标准化层)
    def _get_normlayer(self, net_norm, shape_feature):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feature[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feature, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.InstanceNorm2d(shape_feature[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feature[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit("Unknown net_norm: %s" % net_norm)

    # Create Convolutional Layer (创建卷积层)
    def _make_layers(self, channel, net_width, net_depth, net_act, net_norm, net_pool, img_size):
        layers = []
        in_channels = channel
        if img_size[0] == 28:
            img_size = (32, 32)
        shape_feature = [channel, img_size[0], img_size[1]]
        for i in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3,
                                 padding=3 if channel == 1 and i == 0 else 1)]
            shape_feature[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feature)]
            layers += [self._get_activation(net_act)]
            if net_pool != 'none':
                layers += [self._get_pool(net_pool)]
                shape_feature[1] = shape_feature[1]//2
                shape_feature[2] = shape_feature[2]//2

        return nn.Sequential(*layers), shape_feature

# ConvNet(卷积神经网络)


class ConvNetGap(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pool, img_size=(32, 32)):
        super(ConvNetGap, self).__init__()

        self.features, shape_feature = self._make_layers(
            channel, net_width, net_depth, net_act, net_norm, net_pool, img_size)
        num_feat = shape_feature[0] * shape_feature[1] * shape_feature[2]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feature[0], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    # Activation Function (激活函数)
    def _get_activation(self, net_act):
        if net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'tanh':
            return nn.Tanh()
        elif net_act == 'softmax':
            return nn.Softmax(dim=1)
        else:
            exit("Unknown Activation Function: %s" % net_act)

    # Pooling Layer (池化层)
    def _get_pool(self, net_pool):
        if net_pool == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pool == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pool == 'globalpooling':
            return nn.AdaptiveAvgPool2d(1)
        elif net_pool == 'none':
            return None
        else:
            exit("Unknown net_pooling: %s" % net_pool)

    # Normalization Layer (标准化层)
    def _get_normlayer(self, net_norm, shape_feature):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feature[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feature, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.InstanceNorm2d(shape_feature[0], shape_feature[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feature[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit("Unknown net_norm: %s" % net_norm)

    # Create Convolutional Layer (创建卷积层)
    def _make_layers(self, channel, net_width, net_depth, net_act, net_norm, net_pool, img_size):
        layers = []
        in_channels = channel
        if img_size[0] == 28:
            img_size = (32, 32)
        shape_feature = [channel, img_size[0], img_size[1]]
        for i in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3,
                                 padding=3 if channel == 1 and i == 0 else 1)]
            shape_feature[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feature)]
            layers += [self._get_activation(net_act)]
            if net_pool != 'none':
                layers += [self._get_pool(net_pool)]
                shape_feature[1] = shape_feature[1]//2
                shape_feature[2] = shape_feature[2]//2

        return nn.Sequential(*layers), shape_feature

# LeNet(卷积神经网络 by Yann LeCun)


class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5,
                      padding=2 if channel == 1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x
    
# AlexNet(卷积神经网络 by Alex Krizhevsky)
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5,
                      padding=2 if channel == 1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x

# AlexNetBN


class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1,
                      padding=4 if channel == 1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# VGG( Visual Geometry Group)
    cfg_vgg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm', res=32):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(VGG.cfg_vgg [vgg_name], norm, res)
        self.classifier = nn.Linear(
            512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm, res):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                           nn.GroupNorm(
                               x, x, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1 if res == 32 else 2)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)

def VGG11_Tiny(channel, num_classes):
    return VGG('VGG11', channel, num_classes, res=64)

def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')

def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)

def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)

def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)

#ResNet_AP
class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)  # modification
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)  # modification
        self.bn2 = nn.GroupNorm(
            planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(
            64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel ==
                                    1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes)
