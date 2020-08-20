import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from modules import VRCNN
import argparse
from data_utils import TrainsetLoader,ValidationsetLoader,TestsetLoader,ycbcr2rgb,mse_weight_loss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model_log import *
import time
import torchvision
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from torchvision.transforms import ToPILImage,ToTensor
from ws_psnr import ws_psnr
from ws_ssim import ws_ssim
from psnr import psnr, calculate_ssim

'''
训练参数设置
'''
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--test_val_batchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')  #暂时不用
parser.add_argument('--gpus', default=4, type=int, help='number of gpu')
parser.add_argument('--val_dataset_hr', type=str, default='/data1/zbr/VR-super-resolution/data/test/VR/GT')
parser.add_argument('--val_dataset_lr', type=str, default='/data1/zbr/VR-super-resolution/data/test/VR/LR')
parser.add_argument('--pre_result', type=str, default='./results',help='model prediction results')
parser.add_argument('--model_save_folder', default='model/final_model/my_final-726dual', help='Location to save checkpoint models')
parser.add_argument('--train_log', type=str,default='train_log')
parser.add_argument('--exp_name', type=str,default='726xr-dual')
parser.add_argument('--test_model', type=str, default='vrcnn_final_epoch_272.pth', help='lr change flag')


opt = parser.parse_args()
gpus_list = range(2,opt.gpus)

def main():
    
	'''
	训练时时并行的，测试时也应当并行，不然会报告如下的错误：
	Missing key(s) in state_dict: ...(如：conv1.weight)
	'''
	print('testing processing....')

	#加载模型
	test_model = VRCNN(opt.upscale_factor)
	test_model = torch.nn.DataParallel(test_model,device_ids=gpus_list,output_device=gpus_list[1])

	test_model = test_model.cuda(gpus_list[0])

	print('---------- Networks architecture -------------')
	print_network(test_model)
	print('----------------------------------------------')

	#加载预训练模型
	model_name = os.path.join(opt.model_save_folder,opt.exp_name,opt.test_model)
	print('model_name=',model_name)
	if os.path.exists(model_name):
		pretrained_dict=torch.load(model_name,map_location=lambda storage, loc: storage)
		model_dict=test_model.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		test_model.load_state_dict(model_dict)
		print('Pre-trained SR model is loaded.')

	if not os.path.exists(opt.pre_result):
		os.mkdir(opt.pre_result)

	with open(opt.train_log + '/psnr_ssim-xr-200.txt', 'a') as psnr_ssim:
		with torch.no_grad():
			ave_psnr = 0
			ave_ssim = 0
			single_ave_psnr = 0
			single_ave_ssim = 0
			numb = 2
			valSet = ValidationsetLoader(opt.val_dataset_hr,opt.val_dataset_lr)
			valLoader = DataLoader(dataset=valSet,batch_size=opt.test_val_batchSize,shuffle=False)
			val_bar = tqdm(valLoader)
			for data in val_bar:
				test_model.eval()
				# dual_net.eval()
				batch_lr_y, label, SR_cb,SR_cr,idx,bicubic_restore = data
				batch_lr_y,label = Variable(batch_lr_y).cuda(gpus_list[0]), Variable(label).cuda(gpus_list[0])
				output = test_model(batch_lr_y)

				SR_ycbcr = np.concatenate((np.array(output.squeeze(0).data.cpu()), SR_cb, SR_cr), axis=0).transpose(1,2,0)            
				SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
				SR_rgb = np.clip(SR_rgb, 0, 255)
				SR_rgb = ToPILImage()(SR_rgb.astype(np.uint8))
				#ToTensor() ---image(0-255)==>image(0-1), (H,W,C)==>(C,H,W)
				SR_rgb = ToTensor()(SR_rgb)

				#将给定的Tensor保存成image文件。如果给定的是mini-batch tensor，那就用make-grid做成雪碧图，再保存。与utils.make_grid()配套使用
				if not os.path.exists(opt.pre_result+'/'+opt.exp_name):
					os.mkdir(opt.pre_result+'/'+opt.exp_name)
				utils.save_image(SR_rgb, opt.pre_result+'/' +opt.exp_name +'/' + 'my'+str(numb).rjust(3,'0')+'.png') 
				numb = numb + 1

				psnr_value =  psnr(np.array(torch.squeeze(label).data.cpu())*255,np.array(torch.squeeze(output).data.cpu())*255)
				ave_psnr = ave_psnr + psnr_value
				single_ave_psnr = single_ave_psnr + psnr_value
				ssim_value =  calculate_ssim(np.array(torch.squeeze(label).data.cpu())*255,np.array(torch.squeeze(output).data.cpu())*255)
				ave_ssim = ave_ssim + ssim_value
				single_ave_ssim = single_ave_ssim + ssim_value
				
				val_bar.set_description('===>{}th video {}th frame, wsPSNR:{:.4f} dB,wsSSIM:{:.6f}'.format(idx // 98 + 1,idx % 98 + 1,psnr_value,ssim_value))
				
				if idx == 293 or idx == 97 or idx == 195 or idx == 391:
					print("===> {}th video Avg. wsPSNR: {:.4f} dB".format(idx // 98+1,single_ave_psnr / 98))
					print("===> {}th video Avg. wsSSIM: {:.6f}".format(idx // 98+1,single_ave_ssim / 98))
					psnr_ssim.write('===>{}th video avg wsPSNR:{:.4f} dB,wsSSIM:{:.6f}\n'.format(idx // 98+1,single_ave_psnr / 98,single_ave_ssim / 98))
					single_ave_psnr = 0
					single_ave_ssim = 0

			print("===> All Avg. wsPSNR: {:.4f} dB".format(ave_psnr / len(valLoader)))
			print("===> ALL Avg. wsSSIM: {:.6f}".format(ave_ssim / len(valLoader)))
			psnr_ssim.write('===>all videos avg wsPSNR:{:.4f} dB,wsSSIM:{:.6f}\n'.format(ave_psnr / len(valLoader),ave_ssim / len(valLoader)))

	print('testing finished!')

if __name__ == '__main__':
    main()
