import numpy as np
import numpy
import scipy.misc
#或者用imageio中的imread进行读取，import imageio          imageio.imread()
import os
import math
import cv2

def psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    print('before mse = ',np.sum((img1 - img2)**2))
    print('mse weight = ',img1.shape[0] * img1.shape[1])
    mse = np.mean((img1 - img2)**2)
    print('mse=',mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ws_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    h,w = img1.shape
    weight = np.zeros((h,w))
    for i in range(h):
        weight[i,:] = math.cos((i-h/2 + 0.5)*(math.pi/h))
    wmse = (1/np.sum(weight))*np.sum((img1 - img2)**2*weight)
    if wmse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(wmse))

def ws_psnr2(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    h,w = img1.shape
    weight = np.zeros((h,w))
    for i in range(1,h//6+1):
        weight[i-1,:] = -((h // 6 - i)*1.0 / (h // 6)) + 1.0         #f(x) = -x + 1
    for i in range(h//6):
        weight[h-1-i,:] = weight[i,:]
    wmse = (1/np.sum(weight))*np.sum((img1 - img2)**2*weight)
    if wmse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(wmse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  #产生一个11维的列向量，其值要符合方差1.5
    #11*11
    window = np.outer(kernel, kernel.transpose())   #用于计算外积，即一个m维数组a和一个n维数组b，外积结果为一个m*n的数组,元素为res[i,j]=a[i]*b[j]

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def cal_psnr(LRdir,HRdir):
    imgs_lr = sorted(os.listdir(LRdir))
    imgs_hr = sorted(os.listdir(HRdir))
    total_psnr = 0
    total_ssim = 0
    for index in range(len(imgs_lr)):
        lr = scipy.misc.imread(os.path.join(LRdir,imgs_lr[index]))
        hr = scipy.misc.imread(os.path.join(HRdir,imgs_hr[index]))   #是Numpy数组类型
        total_psnr = total_psnr + psnr(lr, hr)
        total_ssim = total_ssim + calculate_ssim(lr,hr)
    print(total_psnr / len(imgs_lr))
    print('-----------------------')
    print(total_ssim / len(imgs_lr))

if __name__ == '__main__':
    cal_psnr('./results/VRx4_n_d_casa_d/','./data/test/VR/lr_x4/')



