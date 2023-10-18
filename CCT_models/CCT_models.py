#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import json
import io

from .encoder import * 
from .decoders import *

with open("parameters/CCT_parameters.json", 'r') as f:
    CCT_arguments = json.load(f)
    device = CCT_arguments["device"]
    device = torch.device(device)


class Model(nn.Module):
    def __init__(self, mode='semi', arguments=CCT_arguments):
        super(Model, self).__init__()

        self.mode = mode
        self.nb_classes = arguments["model"]["nb_classes"]

        # encoder 
        self.encoder = Encoder(arguments=arguments).to(device)

        # decoders
        self.upscale = arguments["model"]["upscale"]
        self.in_channels_dec = self.encoder.in_channels_psp // 4

        self.main_decoder = MainDecoder(self.upscale, self.in_channels_dec, self.nb_classes).to(device)

        if self.mode == "semi":
            drop_decoder = [DropOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes,drop_rate=arguments["model"]['drop_rate'], spatial_dropout=arguments["model"]['spacial_dropout']).to(device) for _ in range(arguments["model"]['DropOutDecoder'])]
            feature_drop = [FeatureDropDecoder(self.upscale, self.in_channels_dec, self.nb_classes).to(device) for _ in range(arguments["model"]['FeatureDropDecoder'])]
            feature_noise = [FeatureNoiseDecoder(self.upscale, self.in_channels_dec, self.nb_classes, uniform_range=arguments["model"]['uniform_range']).to(device) for _ in range(arguments["model"]['FeatureNoiseDecoder'])]
            vat_decoder = [VATDecoder(self.upscale, self.in_channels_dec, self.nb_classes, xi=arguments["model"]['xi'],eps=arguments["model"]['eps']).to(device) for _ in range(arguments["model"]['VATDecoder'])]
            cut_decoder = [CutOutDecoder(self.upscale, self.in_channels_dec, self.nb_classes, erase=arguments["model"]['erase']).to(device) for _ in range(arguments["model"]['CutOutDecoder'])]
            context_m_decoder = [ContextMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes).to(device) for _ in range(arguments["model"]['ContextMaskingDecoder'])]
            object_masking = [ObjectMaskingDecoder(self.upscale, self.in_channels_dec, self.nb_classes).to(device) for _ in range(arguments["model"]['ObjectMaskingDecoder'])]

            self.aux_decoders = nn.ModuleList([*drop_decoder, *feature_drop, *feature_noise, *vat_decoder, *cut_decoder, *context_m_decoder, *object_masking])

        if arguments["model"]["pretrained"] == True:
            self.pretraining = arguments["model"]["pretraining"]
            self.load_model(arguments)
            if arguments["model"]["freeze"] != "none":
                self.freeze_weights(arguments)

    
    def load_model(self, arguments):  # for test: missing_keys, unexpected_keys =   

        with open(arguments["model"]["model_path"], 'rb') as f:
            buffer = io.BytesIO(f.read())
            state_dict_a = torch.load(buffer)
            state_dict_b = state_dict_a.copy()

        dic_keys = list(state_dict_b.keys())

        if self.pretraining == "backbone":
            for layer_name in dic_keys:
                new_layer_name = layer_name.replace("encoder.base.", "")
                state_dict_b[new_layer_name] = state_dict_b[layer_name]
                del state_dict_b[layer_name]
            self.encoder.base.load_state_dict(state_dict_b, strict=False)

        if self.pretraining == "full":
            for layer_name in dic_keys:
                new_layer_name = layer_name.replace("main_decoder.", "")
                new_layer_name = new_layer_name.replace("encoder.", "")
                state_dict_b[new_layer_name] = state_dict_b[layer_name]
                del state_dict_b[layer_name]
            self.encoder.load_state_dict(state_dict_b, strict=False)   
            self.main_decoder.load_state_dict(state_dict_b, strict=False)
            if self.mode == "semi":
                for decoder in self.aux_decoders:
                    decoder.load_state_dict(state_dict_b, strict=False)

        print("pretraining ok : ", self.pretraining)


    def freeze_weights(self, arguments):
        assert arguments["model"]["pretrained"] == True, "We can't freeze if weights are randomly initialized"

        if arguments["model"]["freeze"] == "encoder":
            for param in self.encoder.get_backbone_params():
                param.requires_grad = False
            for param in self.encoder.get_module_params():
                param.requires_grad = False
        if arguments["model"]["freeze"] == "backbone":
            for param in self.encoder.get_backbone_params():
                param.requires_grad = False
        print("freezing ok : ", arguments["model"]["freeze"])



    def forward(self, x_l, x_ul=None, eval=False):

        output_l = self.main_decoder(self.encoder(x_l)).to(device)
        if output_l.shape != x_l.shape:
            output_l = F.interpolate(output_l, size=(x_l.size(2), x_l.size(3)), mode='bilinear', align_corners=True)

        if self.mode == 'super':
            return {"output_l" :  output_l}
        
        if eval:
            return {"output_l" :  output_l}
        
        elif self.mode == 'semi':
            assert x_ul is not None 
            # Prediction by main decoder 
            inter_ul = self.encoder(x_ul).to(device)
            output_ul = self.main_decoder(inter_ul).to(device)

            # Prediction by auxiliary decoders
            aux_outputs_ul = [aux_decoder(inter_ul, output_ul.detach()).to(device) for aux_decoder in self.aux_decoders]
            aux_outputs_ul = [F.interpolate(output, size=(x_ul.size(2), x_ul.size(3)), mode='bilinear', align_corners=True) for output in aux_outputs_ul if output.shape != x_ul.shape]

            output_ul = F.interpolate(output_ul, size=(x_ul.size(2), x_ul.size(3)), mode='bilinear', align_corners=True)

            return {"output_l" : output_l, "output_ul" : output_ul, "aux_outputs_ul" : aux_outputs_ul}
        
    def get_backbone_params(self):
        return self.encoder.get_backbone_params()
    
    def get_psp_params(self):
        return self.encoder.get_module_params()
    
    def get_maindecoder_params(self):
        return self.main_decoder.parameters()

    def get_other_params(self):
        if self.mode == 'semi':
            return itertools.chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                        self.aux_decoders.parameters())

        return itertools.chain(self.encoder.get_module_params(), self.main_decoder.parameters())
