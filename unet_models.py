import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.nb_channels = self.params['nb_channels']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['nb_classes']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.nb_channels, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.nb_channels = self.params['nb_channels']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['nb_classes']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output
    


class UNet(nn.Module):
    def __init__(self, arguments):
        super(UNet, self).__init__()

        params = {'nb_channels': arguments["model"]["in_channels"],
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'nb_classes': arguments["model"]["nb_classes"],
                  'bilinear': arguments["model"]["bilinear"],
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, eval=False):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output



class ssl_UNet(nn.Module):
    def __init__(self, arguments):
        super(ssl_UNet, self).__init__()

        params = {'nb_channels': arguments["model"]["in_channels"],
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'nb_classes': arguments["model"]["nb_classes"],
                  'bilinear': arguments["model"]["bilinear"],
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)

        self.aux_decoder1 = Decoder(params)  # dropout decoder
        self.aux_decoder2 = Decoder(params)  # feature dropout decoder
        self.aux_decoder3 = Decoder(params)  # feature noise decoder

        # self.aux_decoder1_1 = Decoder(params)  # dropout decoder
        # self.aux_decoder1_2 = Decoder(params)  # feature dropout decoder
        # self.aux_decoder1_3 = Decoder(params)  # feature noise decoder

        # self.aux_decoder2_1 = Decoder(params)  # dropout decoder
        # self.aux_decoder2_2 = Decoder(params)  # feature dropout decoder
        # self.aux_decoder2_3 = Decoder(params)  # feature noise decoder

        # self.aux_decoder3_1 = Decoder(params)  # dropout decoder
        # self.aux_decoder3_2 = Decoder(params)  # feature dropout decoder
        # self.aux_decoder3_3 = Decoder(params)  # feature noise decoder

        # self.aux_decoder4_1 = Decoder(params)  # dropout decoder
        # self.aux_decoder4_2 = Decoder(params)  # feature dropout decoder
        # self.aux_decoder4_3 = Decoder(params)  # feature noise decoder

    def forward(self, x_l, x_ul=None, eval=False):
        feature_l = self.encoder(x_l)
        logits_l = self.main_decoder(feature_l)

        if eval:
            return logits_l
        
        feature_ul = self.encoder(x_ul)
        main_logits_ul = self.main_decoder(feature_ul)

        aux_logits_ul = []

        aux1_feature = [Dropout(i) for i in feature_ul] 
        aux_logits1 = self.aux_decoder1(aux1_feature)
        aux_logits_ul.append(aux_logits1)
        aux2_feature = [FeatureDropout(i) for i in feature_ul]
        aux_logits2 = self.aux_decoder2(aux2_feature)
        aux_logits_ul.append(aux_logits2)
        aux3_feature = [FeatureNoise(i) for i in feature_ul]
        aux_logits3 = self.aux_decoder3(aux3_feature)
        aux_logits_ul.append(aux_logits3)

        # aux1_1_feature = [Dropout(i) for i in feature_ul] 
        # aux_logits1_1 = self.aux_decoder1_1(aux1_1_feature)
        # aux_logits_ul.append(aux_logits1_1)
        # aux1_2_feature = [FeatureDropout(i) for i in feature_ul]
        # aux_logits1_2 = self.aux_decoder1_2(aux1_2_feature)
        # aux_logits_ul.append(aux_logits1_2)
        # aux1_3_feature = [FeatureNoise(i) for i in feature_ul]
        # aux_logits1_3 = self.aux_decoder1_3(aux1_3_feature)
        # aux_logits_ul.append(aux_logits1_3)

        # aux2_1_feature = [Dropout(i) for i in feature_ul] 
        # aux_logits2_1 = self.aux_decoder2_1(aux2_1_feature)
        # aux_logits_ul.append(aux_logits2_1)
        # aux2_2_feature = [FeatureDropout(i) for i in feature_ul]
        # aux_logits2_2 = self.aux_decoder2_2(aux2_2_feature)
        # aux_logits_ul.append(aux_logits2_2)
        # aux2_3_feature = [FeatureNoise(i) for i in feature_ul]
        # aux_logits2_3 = self.aux_decoder2_3(aux2_3_feature)
        # aux_logits_ul.append(aux_logits2_3)

        # aux3_1_feature = [Dropout(i) for i in feature_ul] 
        # aux_logits3_1 = self.aux_decoder3_1(aux3_1_feature)
        # aux_logits_ul.append(aux_logits3_1)
        # aux3_2_feature = [FeatureDropout(i) for i in feature_ul]
        # aux_logits3_2 = self.aux_decoder3_2(aux3_2_feature)
        # aux_logits_ul.append(aux_logits3_2)
        # aux3_3_feature = [FeatureNoise(i) for i in feature_ul]
        # aux_logits3_3 = self.aux_decoder3_3(aux3_3_feature)
        # aux_logits_ul.append(aux_logits3_3)

        # aux4_1_feature = [Dropout(i) for i in feature_ul] 
        # aux_logits4_1 = self.aux_decoder4_1(aux4_1_feature)
        # aux_logits_ul.append(aux_logits4_1)
        # aux4_2_feature = [FeatureDropout(i) for i in feature_ul]
        # aux_logits4_2 = self.aux_decoder4_2(aux4_2_feature)
        # aux_logits_ul.append(aux_logits4_2)
        # aux4_3_feature = [FeatureNoise(i) for i in feature_ul]
        # aux_logits4_3 = self.aux_decoder4_3(aux4_3_feature)
        # aux_logits_ul.append(aux_logits4_3)

        return logits_l, main_logits_ul, aux_logits_ul
    


def Dropout(x, drop_rate=0.3, spatial_dropout=True):
    drop_out = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(p=drop_rate)
    return drop_out(x)


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    return x.mul(drop_mask)


def FeatureNoise(x, uniform_range=0.3):
    noise_vector = 2*uniform_range*torch.rand(x.shape, device=x.device) - uniform_range
    x_noise = x.mul(noise_vector) + x
    return x_noise