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

class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=128, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.RB_L1_1 = basic_block()
        self.RB_L1_2 = basic_block()
        self.RB_L1_3 = basic_block()
        self.RB_L1_4 = basic_block()
        self.RB_L1_5 = basic_block()
        self.RB_L2_1 = basic_block()
        self.RB_L2_2 = basic_block()
        self.RB_L3_1 = basic_block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        #self.dbpn = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=7, scale_factor=upscale_factor).load_state_dict(torch.load('DBPN_x4.pth', map_location=lambda storage, loc: storage))

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        if self.HR_in:
            base = x_center
        else:
            #base = self.dbpn(x_center)
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, supp_feature, ref_feature):
        x = supp_feature  # b,c,h,w
        y = ref_feature

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)

        g_y = g_y.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_y)

        f_div_C = F.softmax(f, dim=1)

        x1 = torch.matmul(f_div_C, g_y)

        x1 = x1.permute(0, 2, 1).contiguous()

        x1 = x1.view(batch_size, self.inter_channels, *supp_feature.size()[2:])
        W_x1 = self.W(x1)
        z = x + W_x1

        return z

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

class dual_network_texture(nn.Module):
    def __init__(self):
        super(dual_network_texture,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv2 = nn.Conv2d(64,1,kernel_size=3,stride=2,padding=1,bias=False)
        # self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1,bias=False)
        self.lrelu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        return out

def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    b, _ , h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda(0)

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1),1)
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border')
    return output

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


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

class OFRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size = 2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.final_upsample = nn.Upsample(scale_factor = upscale_factor, mode='bilinear')
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.upscale_factor = upscale_factor
        self.is_training = is_training
        # Level 1
        self.conv_L1_1 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)
        self.RDB1_1 = RDB(4, 32, 32)
        self.RDB1_2 = RDB(4, 32, 32)
        self.bottleneck_L1 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L1_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        # Level 2
        self.conv_L2_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
        self.RDB2_1 = RDB(4, 32, 32)
        self.RDB2_2 = RDB(4, 32, 32)
        self.bottleneck_L2 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L2_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        # Level 3
        self.conv_L3_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)
        self.RDB3_1 = RDB(4, 32, 32)
        self.RDB3_2 = RDB(4, 32, 32)
        self.bottleneck_L3 = nn.Conv2d(64, 2*upscale_factor**2, 3, 1, 1, bias=False)
        self.conv_L3_2 = nn.Conv2d(2*upscale_factor**2, 2*upscale_factor**2, 3, 1, 1, bias=True)
    def forward(self, x):
        # Level 1
        x_L1 = self.pool(x)
        _, _, h, w = x_L1.size()
        input_L1 = self.conv_L1_1(x_L1)
        buffer_1 = self.RDB1_1(input_L1)
        buffer_2 = self.RDB1_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L1 = self.bottleneck_L1(buffer)
        optical_flow_L1 = self.conv_L1_2(optical_flow_L1)
        optical_flow_L1_upscaled = self.upsample(optical_flow_L1) # *2
        if self.is_training is True:
            x_L1_res = flow_warp(torch.unsqueeze(x_L1[:, 0, :, :], dim=1), optical_flow_L1) - torch.unsqueeze(x_L1[:, 1, :, :], dim=1)
        # Level 2
        x_L2 = flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_L1_upscaled)
        x_L2_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L2
        x_L2 = torch.cat((x, x_L2, x_L2_res,optical_flow_L1_upscaled), 1)
        input_L2 = self.conv_L2_1(x_L2)
        buffer_1 = self.RDB2_1(input_L2)
        buffer_2 = self.RDB2_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L2 = self.bottleneck_L2(buffer)
        optical_flow_L2 = self.conv_L2_2(optical_flow_L2)
        optical_flow_L2 = optical_flow_L2 + optical_flow_L1_upscaled
        if self.is_training is True:
            x_L2_res = flow_warp(torch.unsqueeze(x_L2[:, 0, :, :], dim=1), optical_flow_L2) - torch.unsqueeze(x_L2[:, 1, :, :], dim=1)
        # Level 3
        x_L3 = flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_L2)
        x_L3_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L3
        x_L3 = torch.cat((x, x_L3, x_L3_res, optical_flow_L2), 1)
        input_L3 = self.conv_L3_1(x_L3)
        buffer_1 = self.RDB3_1(input_L3)
        buffer_2 = self.RDB3_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L3 = self.bottleneck_L3(buffer)
        optical_flow_L3 = self.conv_L3_2(optical_flow_L3)
        optical_flow_L3 = self.shuffle(optical_flow_L3) + self.final_upsample(optical_flow_L2) # *4
        if self.is_training is False:
            return optical_flow_L3
        if self.is_training is True:
            return x_L1_res, x_L2_res, optical_flow_L1, optical_flow_L2, optical_flow_L3

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

#用于计算感知损失
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)    #这个值具体数据具体分析
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VRCNN.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    # if gpu_ids:
    #     netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF

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
        self.non_local = NonLocalBlock(nf,nf//2)
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
    
    def NonLocalOps(self, flames_feature):
        feature = []
        batch_size, num, ch, h, w = flames_feature.size()
        ref_feature = flames_feature[:, self.center, :, :, :].clone()
        for i in range(num):
            if i == num // 2:
                feature.append(ref_feature)
                continue
            supp_feature = flames_feature[:, i, :, :, :].contiguous()
            fea = self.non_local(supp_feature, ref_feature)

            feature.append(fea)
        return feature
    
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

        # sa_ca_fea = self.sa_ca(L1_fea.view(B,-1,H,W))
        sa_ca_fea = self.sa_ca(aligned_fea.view(B,-1,H,W))
        output_attention =torch.mul(sa_ca_fea,aligned_fea.view(B,-1,H,W))

        # output_attention =torch.mul(sa_ca_fea,L1_fea.view(B,-1,H,W))
        # output_saca = output_attention + L1_fea.view(B,-1,H,W)

        output_saca = output_attention + aligned_fea.view(B,-1,H,W)

        # res_output = self.SRnet(aligned_fea.view(B,-1,H,W))
        res_output = self.SRnet(output_saca)
        # base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        # output = res_output + base
        sisr = self.dbpn(x[:,1,:,:,:])
        output = res_output + sisr
        # output = res_output 
        output = self.conv_final1(output)
        output = self.conv_final2(output)
        output = self.conv_final3(output)
        # sisr = self.dbpn(x_center)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        output += base
        # output = output + sisr
        return output

class VRCNNattention(nn.Module):
    def __init__(self,upscale_factor, is_training=False,center=None,nf=64, nframes=3):
    # def __init__(self,upscale_factor, is_training=False):
        super(VRCNNattention, self).__init__()
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
        # self.sa_ca = spatial_channel_attention()

    
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


        res_output = self.SRnet(aligned_fea.view(B,-1,H,W))
        sisr = self.dbpn(x[:,1,:,:,:])
        output = res_output + sisr
        output = self.conv_final1(output)
        output = self.conv_final2(output)
        output = self.conv_final3(output)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        output += base
        return output

class VRCNNalignment(nn.Module):
    def __init__(self,upscale_factor, is_training=False,center=None,nf=64, nframes=3):
    # def __init__(self,upscale_factor, is_training=False):
        super(VRCNNalignment, self).__init__()
        self.upscale_factor = upscale_factor
        self.center = nframes // 2 if center is None else center
        self.is_training = is_training
        # self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.conv1 = nn.Conv2d(1, nf, 3, 1, 1, bias = True)
         #functools.partial(a,b,...) 固定函数a中某些参数的值(从左到右顺序固定)，然后返回一个新的函数
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 3)
        # self.pcd_align = PCD_Align()
        self.non_local = NonLocalBlock(nf,nf//2)
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

        L1_fea = L1_fea.view(B, N, -1, H, W)    #view()函数类似于reshape()函数的作用

        sa_ca_fea = self.sa_ca(L1_fea.view(B,-1,H,W))

        output_attention =torch.mul(sa_ca_fea,L1_fea.view(B,-1,H,W))
        output_saca = output_attention + L1_fea.view(B,-1,H,W)

        res_output = self.SRnet(output_saca)
        sisr = self.dbpn(x[:,1,:,:,:])
        output = res_output + sisr
        output = self.conv_final1(output)
        output = self.conv_final2(output)
        output = self.conv_final3(output)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        output += base
        return output
        
class VRCNNPost(nn.Module):
    def __init__(self,upscale_factor, is_training=False,center=None,nf=64, nframes=3):
    # def __init__(self,upscale_factor, is_training=False):
        super(VRCNNPost, self).__init__()
        self.upscale_factor = upscale_factor
        self.center = nframes // 2 if center is None else center
        self.is_training = is_training
        # self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.conv1 = nn.Conv2d(1, nf, 3, 1, 1, bias = True)
         #functools.partial(a,b,...) 固定函数a中某些参数的值(从左到右顺序固定)，然后返回一个新的函数
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 3)
        self.pcd_align = PCD_Align()
        self.non_local = NonLocalBlock(nf,nf//2)
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
        return output
        
class VRCNNsfn(nn.Module):
    def __init__(self,upscale_factor, is_training=False,center=None,nf=64, nframes=3):
    # def __init__(self,upscale_factor, is_training=False):
        super(VRCNNsfn, self).__init__()
        self.upscale_factor = upscale_factor
        self.center = nframes // 2 if center is None else center
        self.is_training = is_training
        # self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.conv1 = nn.Conv2d(1, nf, 3, 1, 1, bias = True)
         #functools.partial(a,b,...) 固定函数a中某些参数的值(从左到右顺序固定)，然后返回一个新的函数
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 3)
        self.pcd_align = PCD_Align()
        self.non_local = NonLocalBlock(nf,nf//2)
        self.SRnet = SRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #减少通道数，从33-3
        self.reduce = nn.Conv2d(33, 3, 3, 1, 1, bias=False)

        #sisr-upsample
        # self.dbpn = DBPN(num_channels=1, base_filter=64,  feat = 256, num_stages=7, scale_factor=upscale_factor)
        self.conv_final1 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.conv_final2 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.conv_final3 = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        
        #注意力
        self.sa_ca = spatial_channel_attention()
    
    def NonLocalOps(self, flames_feature):
        feature = []
        batch_size, num, ch, h, w = flames_feature.size()
        ref_feature = flames_feature[:, self.center, :, :, :].clone()
        for i in range(num):
            if i == num // 2:
                feature.append(ref_feature)
                continue
            supp_feature = flames_feature[:, i, :, :, :].contiguous()
            fea = self.non_local(supp_feature, ref_feature)

            feature.append(fea)
        return feature
    
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

        # sa_ca_fea = self.sa_ca(L1_fea.view(B,-1,H,W))
        sa_ca_fea = self.sa_ca(aligned_fea.view(B,-1,H,W))
        output_attention =torch.mul(sa_ca_fea,aligned_fea.view(B,-1,H,W))

        # output_attention =torch.mul(sa_ca_fea,L1_fea.view(B,-1,H,W))
        # output_saca = output_attention + L1_fea.view(B,-1,H,W)

        output_saca = output_attention + aligned_fea.view(B,-1,H,W)

        # res_output = self.SRnet(aligned_fea.view(B,-1,H,W))
        res_output = self.SRnet(output_saca)
        # base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        # output = res_output + base

        # sisr = self.dbpn(x[:,1,:,:,:])
        # output = res_output + sisr
        output = res_output 
        output = self.conv_final1(output)
        output = self.conv_final2(output)
        output = self.conv_final3(output)
        # sisr = self.dbpn(x_center)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        output += base
        # output = output + sisr
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

