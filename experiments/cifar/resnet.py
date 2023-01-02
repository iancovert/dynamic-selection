'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    
Borrowed from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

########## Custom modules and functions ##########

class UpsamplingBlock(nn.Module):
    '''Custom residual block for performing upsampling.'''
    expansion = 1
    
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                        in_planes, self.expansion*planes, kernel_size=2,
                        stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

def make_layer(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
        if stride >= 1:
            layers.append(block(in_planes, planes, stride))
        else:
            layers.append(UpsamplingBlock(in_planes, planes))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)


def ResNet18Backbone():
    '''ResNet18 backbone, including modules up to (batch, 128, 8, 8) tensor.'''
    return nn.Sequential(
        # Initial conv-bn-relu sequence.
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        # Block 1.
        # make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=2, stride=1),
        make_layer(BasicBlock, in_planes=64, planes=64, num_blocks=1, stride=1),

        # Block 2.
        # make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=2, stride=2),
        make_layer(BasicBlock, in_planes=64, planes=128, num_blocks=1, stride=2),
        
        # Block 3.
        # make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=2, stride=2)
        make_layer(BasicBlock, in_planes=128, planes=256, num_blocks=1, stride=2),
        
        # Block 4.
        make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=1, stride=2)
    )

def ResNet18ClassifierHead(num_classes=10):
    '''ResNet18 classifier head, including residual block, GAP and FC layer.'''
    return nn.Sequential(
        # # Block 4.
        # make_layer(BasicBlock, in_planes=256, planes=512, num_blocks=2, stride=2),
        
        # GAP + FC.
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )

def ResNet18SelectorHead():
    '''Custom upsampling module to select 2x2 patches (assuming 32x32 input).'''
    return nn.Sequential(
        # Upsampling block.
        # make_layer(BasicBlock, in_planes=256, planes=128, num_blocks=2, stride=0.5),
        make_layer(BasicBlock, in_planes=512, planes=256, num_blocks=2, stride=0.5),
        
        # Output selections.
        # nn.Conv2d(128, 1, 1)
        nn.Conv2d(256, 1, 1)
    )

