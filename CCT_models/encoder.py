#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from .resnet import *

with open("parameters/CCT_parameters.json", 'r') as f:
    CCT_arguments = json.load(f)
    device = CCT_arguments["device"]
    device = torch.device(device)

class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
    

class Encoder(nn.Module):
    def __init__(self, arguments=CCT_arguments):
        super(Encoder, self).__init__()

        self.nb_RNlayers = arguments["model"]["nb_RNlayers"]
        self.in_channels = arguments["model"]["in_channels"]
        self.pretrained = arguments["model"]["pretrained"]

        if self.nb_RNlayers in [50, 101, 152]: self.in_channels_psp = 2048
        elif self.nb_RNlayers in [18, 34]: self.in_channels_psp = 512
        else: raise ValueError("invalid nb_RNlayers")

        model = resnet_bbs[self.nb_RNlayers](arguments)
        
        self.base = nn.Sequential(
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)
        
        self.psp = _PSPModule(self.in_channels_psp, bin_sizes=[1, 2, 3, 6])

    def forward(self, x):
        x = self.base(x)
        x = self.psp(x)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()