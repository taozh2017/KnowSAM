from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


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
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        # self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
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
        # return x4


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
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
    # def forward(self, x0, x1, x2, x3, x4):
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
        return output, x
        # return output
       
        
class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


class box_decoder_embedding(nn.Module):
    def __init__(self, params):
        super(box_decoder_embedding, self).__init__()

        self.sam_box_embedding_channels = 256
        self.box_nums = params['box_nums']
        f_channel = params['feature_chns'][-1]

        self.conv_1 = ConvBlock(in_channels=params['feature_chns'][-1], out_channels=params['feature_chns'][-1],
                                dropout_p=0.1)
        self.down_1 = DownBlock(params['feature_chns'][-1], params['feature_chns'][-1], 0.1)
        self.conv_2 = ConvBlock(in_channels=params['feature_chns'][-1], out_channels=params['feature_chns'][-1],
                                dropout_p=0.1)
        self.down_2 = DownBlock(params['feature_chns'][-1], params['feature_chns'][-1], 0.1)
        self.conv_3 = nn.Conv2d(params['feature_chns'][-1], params['feature_chns'][-1], kernel_size=1)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.ffn = FeedForward(in_features=params['feature_chns'][-1], hidden_features=self.box_nums * f_channel * 4,
                               out_features=self.box_nums * self.sam_box_embedding_channels * 2, dropout=0.1)

    def forward(self, feature):
        b = feature.shape[0]
        x = self.conv_1(feature)
        x = self.down_1(x)
        x = self.conv_2(x)
        x = self.down_2(x)
        x = self.conv_3(x)
        x = self.global_pooling(x).view(b, -1)
        boxes_embedding = self.ffn(x)

        return boxes_embedding.view(b, self.box_nums * 2, self.sam_box_embedding_channels)
    

class Super_Prompt(nn.Module):
    def __init__(self, in_chns, class_num, point_nums=5, box_nums=1):
        super(Super_Prompt, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'point_nums': point_nums,
                  'box_nums': box_nums
                  }
        self.class_num = class_num
        self.box_decoder = nn.ModuleList()
        for _ in range(class_num):
            self.box_decoder.append(box_decoder_embedding(params))

    def forward(self, x):
        feature = x
        boxes_embedding = []
        for i in range(self.class_num):
            boxes_embedding.append(self.box_decoder[i](feature))
        return None, boxes_embedding, None
    



