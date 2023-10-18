#%% 
import torch
import torch.nn as nn
import json

with open("parameters/CCT_parameters.json", 'r') as f:
    CCT_arguments = json.load(f)
    device = CCT_arguments["device"]
    device = torch.device(device)

# residual block for the resnet 18 and 34 
class ResidualBlock_2sl(nn.Module): 
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None, dilation=1):
        super(ResidualBlock_2sl, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

# residual block for the resnet 50, 101 and 152 
class ResidualBlock_3sl(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(ResidualBlock_3sl, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion) 

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, arguments=CCT_arguments):
        super(ResNet, self).__init__()
        self.inplanes = 64
        multi_grid = arguments["model"]["multi_grid"]
        dilate_scale = arguments["model"]["dilate_scale"]
        isDilation = arguments["model"]["isDilation"]
        nb_classes = arguments["model"]["nb_classes"]
        in_channels = arguments["model"]["in_channels"]

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if isDilation:
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=multi_grid[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=dilate_scale // multi_grid[1])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=dilate_scale // multi_grid[2])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=dilate_scale)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
            self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.fc = nn.Linear(512*block.expansion, nb_classes)

    def _make_layer(self, block, planes, nb_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:  # if the dim of the residual does no match the dim of the output
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes*block.expansion
        for i in range(1, nb_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    

class ResnetBackbone(nn.Module):
    def __init__(self, orig_resnet, arguments):
        super(ResnetBackbone, self).__init__()

        # Take resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = nn.ReLU()
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        tuple_features = list()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features
    
def ResNet18_bb(arguments):
    return ResnetBackbone(ResNet(ResidualBlock_2sl, [3,2,2,2]), arguments)
    # return ResNet(ResidualBlock_2sl, [3,2,2,2])

def ResNet34_bb(arguments):
    return ResnetBackbone(ResNet(ResidualBlock_2sl, [3,4,6,3]), arguments)
    # return ResNet(ResidualBlock_2sl, [3,4,6,3])

def ResNet50_bb(arguments):
    return ResnetBackbone(ResNet(ResidualBlock_3sl, [3,4,6,3]), arguments)
    # return ResNet(ResidualBlock_3sl, [3,4,6,3])

def ResNet101_bb(arguments):
    return ResnetBackbone(ResNet(ResidualBlock_3sl, [3,4,23,3]), arguments)
    # return ResNet(ResidualBlock_3sl, [3,4,23,3])

def ResNet152_bb(arguments):
    return ResnetBackbone(ResNet(ResidualBlock_3sl, [3,8,36,3]), arguments)
    # return ResNet(ResidualBlock_3sl, [3,8,36,3])

resnet_bbs = {18 : ResNet18_bb, 
           34 : ResNet34_bb, 
           50 : ResNet50_bb, 
           101 : ResNet101_bb,
           152 : ResNet152_bb
           }

