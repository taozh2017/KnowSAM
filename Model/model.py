from Model.unet import DownBlock, UpBlock
from Model.vnet import DownsamplingConvBlock, UpsamplingDeconvBlock
from Model.vnet import ConvBlock as vnet_ConvBlock
from Model.unet import ConvBlock as unet_ConvBlock
from Model.discriminator import Discriminator
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)

        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        project_out = not (num_heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, p1, p2):
        q_p1 = self.w_q(p1)
        k_p2 = self.w_k(p2)
        v_p2 = self.w_v(p2)
        q_p1 = rearrange(q_p1, 'b n (h d) -> b h n d', h=self.num_heads)
        k_p2 = rearrange(k_p2, 'b n (h d) -> b h n d', h=self.num_heads)
        v_p2 = rearrange(v_p2, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_p1p2 = einsum('b h i d, b h j d -> b h i j', q_p1, k_p2) * self.scale
        attn_p1p2 = attn_p1p2.softmax(dim=-1)
        # show(attn_p1p2)
        attn_p1p2 = einsum('b h i j, b h j d -> b h i d', attn_p1p2, v_p2)
        attn_p1p2 = rearrange(attn_p1p2, 'b h n d -> b n (h d)')

        attn_p1p2 = self.to_out(attn_p1p2)
        return attn_p1p2


class Cross_Attention_block(nn.Module):
    def __init__(self, input_size, in_channels, patch_size=16, num_heads=8, channel_attn_drop=0.1, pos_embed=True, dim=96, dim_head=64, hid_dim=384):
        super(Cross_Attention_block, self).__init__()
        self.patch_size = patch_size
        input_size = int(input_size)
        assert input_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (input_size // patch_size) ** 2

        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dropout = nn.Dropout(channel_attn_drop)

        self.attn = Attention(dim, num_heads, dim_head, channel_attn_drop)

        if pos_embed:
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim))
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        else:
            self.pos_embed = None

        self.MLP = FeedForward(dim, hid_dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.Dropout(channel_attn_drop),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size // patch_size), w=(input_size // patch_size),  p1=patch_size, p2=patch_size),
        )

    def forward(self, p1, p2):

        p1 = self.to_patch_embedding(p1)
        p2 = self.to_patch_embedding(p2)
        _, n, _ = p1.shape       # n表示每个块的空间分辨率

        if self.pos_embed is not None:
            p1 = p1 + self.pos_embed
            p2 = p2 + self.pos_embed
        p1 = self.dropout(p1)
        p2 = self.dropout(p2)

        attn_p1p2 = self.attn(p1, p2)

        attn_p1p2 = self.MLP(attn_p1p2) + attn_p1p2
        attn_p1p2 = self.to_out(attn_p1p2)
        # show_attn(attn_p1p2)
        return attn_p1p2


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.block_one = vnet_ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = vnet_ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = vnet_ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = vnet_ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = vnet_ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = vnet_ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = vnet_ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = vnet_ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = vnet_ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_chns, class_num, bilinear):
        super(UNet, self).__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': bilinear,
                  'acti_func': 'relu',
                  }
        self.params = params
        self.n_class = self.params['class_num']
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.in_conv = unet_ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

class KnowSAM(nn.Module):
    def __init__(self, args, bilinear=False, has_dropout=False):
        super(KnowSAM, self).__init__()
        self.has_dropout = has_dropout
        self.UNet = UNet(in_chns=args.in_channels, class_num=args.num_classes, bilinear=bilinear)
        self.VNet = VNet(n_channels=args.in_channels, n_classes=args.num_classes)
        self.Discriminator = Discriminator(in_channels=args.num_classes, out_conv_channels=args.num_classes)

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def forward(self, x):
        x0_u = self.UNet.in_conv(x)
        x0_v = self.VNet.block_one(x)

        x1_u = self.UNet.down1(x0_u)
        x1_v = self.VNet.block_one_dw(x0_v)

        x2_u = self.UNet.down2(x1_u)
        x2_v = self.VNet.block_two_dw(self.VNet.block_two(x1_v))

        x3_u = self.UNet.down3(x2_u)
        x3_v = self.VNet.block_three_dw(self.VNet.block_three(x2_v))

        x4_u = self.UNet.down4(x3_u)
        x4_v = self.VNet.block_four_dw(self.VNet.block_four(x3_v))

        # unet decoder
        x_u = self.UNet.up1(x4_u, x3_u)
        x_u = self.UNet.up2(x_u, x2_u)
        x_u = self.UNet.up3(x_u, x1_u)
        x_u = self.UNet.up4(x_u, x0_u)
        pred_UNet = self.UNet.out_conv(x_u)

        # vnet_decoder
        x_v = self.VNet.block_five_up(self.VNet.block_five(x4_v))
        x_v = x_v + x3_v
        x_v = self.VNet.block_six_up(self.VNet.block_six(x_v))
        x_v = x_v + x2_v
        x_v = self.VNet.block_seven_up(self.VNet.block_seven(x_v))
        x_v = x_v + x1_v
        x_v = self.VNet.block_eight_up(self.VNet.block_eight(x_v))
        x_v = x_v + x0_v
        x_v = self.VNet.block_nine(x_v)
        if self.has_dropout:
            x_v = self.VNet.dropout(x_v)
        pred_VNet = self.VNet.out_conv(x_v)

        pred_UNet_soft = torch.softmax(pred_UNet, dim=1)
        pred_VNet_soft = torch.softmax(pred_VNet, dim=1)

        entmap1 = self.get_entropy_map(pred_UNet_soft)
        entmap2 = self.get_entropy_map(pred_VNet_soft)

        fusion_map = self.Discriminator(pred_UNet, pred_VNet, pred_UNet_soft,  pred_VNet_soft, entmap1, entmap2)
        return pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map



    





    

