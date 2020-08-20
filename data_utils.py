import numpy as np
from PIL import Image,ImageOps
import os
import torch
from torch.utils.data.dataset import Dataset
import math
import random
import torchvision
from torchvision.transforms import Resize
from skimage import feature
import torchvision.utils as utils

class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class mse_weight_loss(torch.nn.Module):

    def __init__(self):
        super(mse_weight_loss, self).__init__()

    def forward(self, x, y):
        equ = torch.zeros((y.size(2),y.size(3))).cuda(0)
        N = y.size(2)
        for j in range(0,y.size(2)):   #hang
            for i in range(0,y.size(3)):  #lie
                val = math.pi/N
                equ[j,i] = math.cos( (j - (N/2) + 0.5) * val )

	    #weight_value = torch.from_numpy(weight_value[np.newaxis,:, :]).cuda(gpus_list[0])
        diff = x - y
        loss = torch.sum((diff * diff )*equ) / torch.sum(equ)
        return loss


class TrainsetLoader(Dataset):
    def __init__(self, trainset_dir_hr, trainset_dir_lr,upscale_factor, patch_size, n_iters):
        super(TrainsetLoader).__init__()
        self.trainset_dir_hr = trainset_dir_hr
        self.trainset_dir_lr = trainset_dir_lr
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.n_iters = n_iters
        self.video_list = os.listdir(trainset_dir_hr)
    def __getitem__(self, idx):
        idx_video = random.randint(0, self.video_list.__len__()-1)
        idx_frame = random.randint(1, 98)
        lr_dir = self.trainset_dir_lr +'/'+ self.video_list[idx_video]
        hr_dir = self.trainset_dir_hr +'/'+ self.video_list[idx_video]
        # read HR & LR frames
        LR0 = Image.open(lr_dir + '/' + str(idx_frame).rjust(3,'0') + '.png')
        LR1 = Image.open(lr_dir + '/' +  str(idx_frame + 1).rjust(3,'0') + '.png')
        LR2 = Image.open(lr_dir + '/' +  str(idx_frame + 2).rjust(3,'0') + '.png')
        HR0 = Image.open(hr_dir + '/' +  str(idx_frame).rjust(3,'0') + '.png')
        HR1 = Image.open(hr_dir + '/' +  str(idx_frame + 1).rjust(3,'0') + '.png')
        HR2 = Image.open(hr_dir + '/' +  str(idx_frame + 2).rjust(3,'0') + '.png')

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0
        # extract Y channel for LR inputs

        HR0 = rgb2y(HR0)
        HR1 = rgb2y(HR1)
        HR2 = rgb2y(HR2)
        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)


        # img = Image.fromarray(np.uint8(LR1_texture))
        # img.save('new1.jpg')
        
        # crop patchs randomly
        HR0, HR1, HR2, LR0, LR1, LR2,start_h,end_h,start_w,end_w = random_crop(HR0, HR1, HR2, LR0, LR1, LR2, self.patch_size, self.upscale_factor)
        HR1_texture = feature.canny(HR1)
        LR1_texture = feature.canny(LR1)
        LR1_texture = torch.from_numpy(LR1_texture)
        LR1_texture = LR1_texture.type(torch.FloatTensor)
        # HR0 = HR0[:, :, np.newaxis]
        # HR1 = HR1[:, :, np.newaxis]
        # HR2 = HR2[:, :, np.newaxis]
        # LR0 = LR0[:, :, np.newaxis]
        # LR1 = LR1[:, :, np.newaxis]
        # LR2 = LR2[:, :, np.newaxis]

        # HR = np.concatenate((HR0, HR1, HR2), axis=2)
        # LR = np.concatenate((LR0, LR1, LR2), axis=2)
        HR0 = HR0[:, :, np.newaxis, np.newaxis]
        HR1 = HR1[:, :, np.newaxis, np.newaxis]
        HR2 = HR2[:, :, np.newaxis, np.newaxis]
        LR0 = LR0[:, :, np.newaxis, np.newaxis]
        LR1 = LR1[:, :, np.newaxis, np.newaxis]
        LR2 = LR2[:, :, np.newaxis, np.newaxis]
        # print(LR1.shape)

        HR = np.concatenate((HR0, HR1, HR2), axis=3)
        LR = np.concatenate((LR0, LR1, LR2), axis=3)
        # data augmentation
        LR, HR = augumentation()(LR, HR)
        return toTensor(LR), toTensor(HR),HR1_texture,LR1_texture
    def __len__(self):
        return self.n_iters

class ValidationsetLoader(Dataset):
    def __init__(self, validate_dir_hr, validate_dir_lr):
        super(TrainsetLoader).__init__()
        self.validate_dir_hr = validate_dir_hr
        self.validate_dir_lr = validate_dir_lr
        self.video_list = sorted(os.listdir(validate_dir_hr))
        self.idx_frame = 0
        self.idx_video = -1   #代表第几个视频
        self.n_iters = 100 * len(self.video_list) - 2 * len(self.video_list)
    def __getitem__(self, idx):

        if idx % 98 == 0:
            self.idx_video = self.idx_video + 1
            self.idx_frame = 0

        self.idx_frame = self.idx_frame + 1
        lr_dir = self.validate_dir_lr +'/'+ self.video_list[self.idx_video]
        hr_dir = self.validate_dir_hr +'/'+ self.video_list[self.idx_video]
        # read HR & LR frames
        LR0 = Image.open(lr_dir + '/' + str(self.idx_frame).rjust(3,'0') + '.png')
        LR1 = Image.open(lr_dir + '/' +  str(self.idx_frame + 1).rjust(3,'0') + '.png')
        LR2 = Image.open(lr_dir + '/' +  str(self.idx_frame + 2).rjust(3,'0') + '.png')
        HR1 = Image.open(hr_dir + '/' +  str(self.idx_frame + 1).rjust(3,'0') + '.png')

        W, H = LR1.size
        LR1_bicubic = LR1.resize((W*4, H*4), Image.BICUBIC)
        LR1_bicubic = np.array(LR1_bicubic, dtype=np.float32) / 255.0
        _, SR_cb, SR_cr = rgb2ycbcr(LR1_bicubic)

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        # extract Y channel for LR inputs

        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)
        HR1,HR1_cb, HR1_cr = rgb2ycbcr(HR1)

        LR0 = LR0[:, :, np.newaxis, np.newaxis]
        LR1 = LR1[:, :, np.newaxis, np.newaxis]
        LR2 = LR2[:, :, np.newaxis, np.newaxis]
        HR = HR1[:, :, np.newaxis, np.newaxis]

        LR = np.concatenate((LR0, LR1, LR2), axis=3)
        return toTensor(LR), toTensor(HR),SR_cb,SR_cr,idx,LR1_bicubic
    def __len__(self):
        return self.n_iters

class TestsetLoader(Dataset):
    def __init__(self, dir_hr, dir_lr):
            super(TestsetLoader).__init__()
            self.dir_hr = dir_hr
            self.dir_lr = dir_lr
            self.video_list = sorted(os.listdir(dir_hr))
            self.idx_frame = 0
            self.idx_video = -1   #代表第几个视频
            self.n_iters = 100 * len(self.video_list) - 2 * len(self.video_list)
    def __getitem__(self, idx):

        if idx % 98 == 0:
            self.idx_video = self.idx_video + 1
            self.idx_frame = 0

        self.idx_frame = self.idx_frame + 1
        lr_dir = self.dir_lr +'/'+ self.video_list[self.idx_video]
        hr_dir = self.dir_hr +'/'+ self.video_list[self.idx_video]
        # read HR & LR frames
        LR0 = Image.open(lr_dir + '/' + str(self.idx_frame).rjust(3,'0') + '.png')
        LR1 = Image.open(lr_dir + '/' +  str(self.idx_frame + 1).rjust(3,'0') + '.png')
        LR2 = Image.open(lr_dir + '/' +  str(self.idx_frame + 2).rjust(3,'0') + '.png')
        HR1 = Image.open(hr_dir + '/' +  str(self.idx_frame + 1).rjust(3,'0') + '.png')
            
        W, H = LR1.size
        LR1_bicubic = LR1.resize((W*4, H*4), Image.BICUBIC)
        LR1_bicubic = np.array(LR1_bicubic, dtype=np.float32) / 255.0
        _, SR_cb, SR_cr = rgb2ycbcr(LR1_bicubic)

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        # extract Y channel for LR inputs

        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)
        HR1,HR1_cb, HR1_cr = rgb2ycbcr(HR1)

        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]
        HR = HR1[:, :, np.newaxis]

        LR = np.concatenate((LR0, LR1, LR2), axis=2)
        return toTensor2(LR),toTensor2(HR),SR_cb,SR_cr,idx,LR1_bicubic
    def __len__(self):
        return self.n_iters

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :,:]
            target = target[:, ::-1,:, :]
        if random.random()<0.5:
            input = input[::-1, :, :,:]
            target = target[::-1, :,:, :]
        if random.random()<0.5:
            input = input.transpose(1, 0, 2,3)
            target = target.transpose(1, 0, 2,3)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)


def random_crop(HR0, HR1, HR2, LR0, LR1, LR2, patch_size_lr, upscale_factor):
    h_hr, w_hr = HR0.shape
    h_lr = h_hr // upscale_factor
    w_lr = w_hr // upscale_factor
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * upscale_factor
    h_end_hr = (idx_h - 1 + patch_size_lr) * upscale_factor
    w_start_hr = (idx_w - 1) * upscale_factor
    w_end_hr = (idx_w - 1 + patch_size_lr) * upscale_factor

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR0[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR1 = HR1[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR2 = HR2[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR0 = LR0[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR1 = LR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR2 = LR2[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return HR0, HR1, HR2, LR0, LR1, LR2,h_start_hr,h_end_hr,w_start_hr,w_end_hr

def toTensor(img):
    img = torch.from_numpy(img.transpose((3,2, 0, 1)))
    img.float().div(255)
    return img

def toTensor2(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img.float().div(255)
    return img

def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr

def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb

def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] +16 / 255.0
    return image_y
