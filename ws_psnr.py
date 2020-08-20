import numpy as np
import math
from scipy.ndimage import imread
import cv2


import numpy as np
import math


def genERP(i,j,N):
    val = math.pi/N
    # w_map[i+j*w] = cos ((j - (h/2) + 0.5) * PI/h)
    w = math.cos( (j - (N/2) + 0.5) * val )
    return w

def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((img.shape[0],img.shape[1]))

    for j in range(0,equ.shape[0]):
        for i in range(0,equ.shape[1]):
            equ[j,i] = genERP(i,j,equ.shape[0])

    return equ

def getGlobalWSMSEValue(mx,my):

    mw = compute_map_ws(mx)
    val = np.sum( np.multiply((mx-my)**2,mw) )
    print('w/o wieight =',np.sum((mx-my)**2))
    print('weight = ',np.sum(mw))
    print('before ws-mse=',val)
    den = val / np.sum(mw)

    return den

def ws_psnr(image1,image2):
    ws_mse   = getGlobalWSMSEValue(image1,image2)
    print('ws-mse=',ws_mse)
    # second estimate the ws_psnr 

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf
    # print("WS-PSNR ",ws_psnr)

    return ws_psnr