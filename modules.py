import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
import functools
import torchvision
from dbpn import Net as DBPN

try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea

class dual_network(nn.Module):
    def __init__(self):
        super(dual_network,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv2 = nn.Conv2d(64,1,kernel_size=3,stride=2,padding=1,bias=False)
        # self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1,bias=False)
        self.lrelu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        return out


class make_dense(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(make_dense, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
    def forward(self, x):
        out = self.leaky_relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nDenselayer, channels, growth):
        super(RDB, self).__init__()
        modules = []
        channels_buffer = channels
        for i in range(nDenselayer):
            modules.append(make_dense(channels_buffer, growth))
            channels_buffer += growth
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(channels_buffer, channels, kernel_size=1, padding=0, bias=False)
        self.weight = nn.Parameter(torch.Tensor([0]))
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # out = out + x
        out = out*self.weight + x
        return out


class SRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(SRnet, self).__init__()
        self.conv = nn.Conv2d(192, 64, 3, 1, 1, bias=False)
        self.RDB_1 = RDB(5, 64, 32)
        self.RDB_2 = RDB(5, 64, 32)
        self.RDB_3 = RDB(5, 64, 32)
        self.RDB_4 = RDB(5, 64, 32)
        self.RDB_5 = RDB(5, 64, 32)
        self.conv_2 = nn.Conv2d(384, 384, 3, 1, 1, bias=True)
        self.conv_3 = nn.Conv2d(384, 384, 3, 1, 1, bias=True)
        self.bottleneck = nn.Conv2d(384, upscale_factor ** 2, 1, 1, 0, bias=False)
        self.conv_4 = nn.Conv2d(upscale_factor ** 2, upscale_factor ** 2, 3, 1, 1, bias=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.last = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.is_training = is_training
        self.lrelu = nn.LeakyReLU(0.1,inplace = True)
        # self.sa_ca = spatial_channel_attention2()
        self.conv_sa_ca = nn.Conv2d(64, 384, 3, 1, 1, bias=True)
    def forward(self, x):
        B,C,H,W = x.size()
        input = self.conv(x)
        buffer_1 = self.RDB_1(input)
        buffer_2 = self.RDB_2(buffer_1)
        buffer_3 = self.RDB_3(buffer_2)
        buffer_4 = self.RDB_4(buffer_3)
        buffer_5 = self.RDB_5(buffer_4)
        output_cat = torch.cat((buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, input), 1)
        # attention = self.sa_ca(output_cat)
        # output_attention =torch.mul(attention,output_cat)
        # output1 = output_attention + self.conv_sa_ca(input)
        output1 = self.conv_2(output_cat)
        # output1 = self.conv_2(output1)
        output1 = self.lrelu(output1)
        output1 = self.conv_3(output1)
        output = self.bottleneck(output1)
        output = self.conv_4(output)
        output = self.shuffle(output)
        output = self.last(output)
        return output
        
class spatial_channel_attention2(nn.Module):
    def __init__(self):
        super(spatial_channel_attention2, self).__init__()
        # self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_c1 = nn.Conv2d(384,384//16,kernel_size=1,bias=False)
        self.conv_c2 = nn.Conv2d(384//16,384,kernel_size=1,bias=False)
        self.conv_s1 = nn.Conv2d(384,384,kernel_size=1,bias=False)
        self.conv_s2 = nn.Conv2d(384,384,kernel_size=1,groups=384,bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ca = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)) 
        ca = self.conv_c1(ca)
        ca = self.lrelu(ca)
        ca = self.conv_c2(ca)

        sa = self.conv_s1(x)
        sa = self.lrelu(sa)
        sa = self.conv_s2(sa)

        output = self.sigmoid(sa + ca)
        return output
        
class spatial_channel_attention(nn.Module):
    def __init__(self):
        super(spatial_channel_attention, self).__init__()
        # self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_c1 = nn.Conv2d(192,192//16,kernel_size=1,bias=False)
        self.conv_c2 = nn.Conv2d(192//16,192,kernel_size=1,bias=False)
        self.conv_s1 = nn.Conv2d(192,192,kernel_size=1,bias=False)
        self.conv_s2 = nn.Conv2d(192,192,kernel_size=1,groups=192,bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        ca = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)) 
        ca = self.conv_c1(ca)
        ca = self.lrelu(ca)
        ca = self.conv_c2(ca)

        sa = self.conv_s1(x)
        sa = self.lrelu(sa)
        sa = self.conv_s2(sa)

        output = self.sigmoid(sa + ca)
        return output

class VRCNN(nn.Module):
    def __init__(self,upscale_factor, is_training=False,center=None,nf=64, nframes=3):
    # def __init__(self,upscale_factor, is_training=False):
        super(VRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.center = nframes // 2 if center is None else center
        self.is_training = is_training
        # self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.conv1 = nn.Conv2d(1, nf, 3, 1, 1, bias = True)
         #functools.partial(a,b,...) 固定函数a中某些参数的值(从左到右顺序固定)，然后返回一个新的函数
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 3)
        self.pcd_align = PCD_Align()
        self.SRnet = SRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #减少通道数，从33-3
        self.reduce = nn.Conv2d(33, 3, 3, 1, 1, bias=False)

        #sisr-upsample
        self.dbpn = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=7, scale_factor=upscale_factor)
        self.conv_final1 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.conv_final2 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.conv_final3 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        
        #注意力
        self.sa_ca = spatial_channel_attention()
    
    
    def forward(self, x):
        
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv1(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)    #view()函数类似于reshape()函数的作用
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        # #### pcd align
        # # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :,:, :].clone(), L2_fea[:, self.center, :,:, :].clone(),
            L3_fea[:, self.center, :,:, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :,:, :].clone(), L2_fea[:, i, :,:, :].clone(),
                L3_fea[:, i, :, :,:].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        sa_ca_fea = self.sa_ca(aligned_fea.view(B,-1,H,W))
        output_attention =torch.mul(sa_ca_fea,aligned_fea.view(B,-1,H,W))


        output_saca = output_attention + aligned_fea.view(B,-1,H,W)

        res_output = self.SRnet(output_saca)

        sisr = self.dbpn(x[:,1,:,:,:])
        output = res_output + sisr

        output = self.conv_final1(output)
        output = self.conv_final2(output)
        output = self.conv_final3(output)

        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        output += base
        return output

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

