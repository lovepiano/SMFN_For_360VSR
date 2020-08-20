import torch
import os

#保存模型
'''
每个epoch保存一个模型
'''
def checkpoint(model,epoch,model_save_folder,model_type):
    if not os.path.exists(model_save_folder):
        os.mkdir(model_save_folder)
    model_out_path = model_save_folder+'/'+ model_type + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))  
    # write_log(opt.train_log,log_file,"Checkpoint saved to {}".format(model_out_path))
    # write_log(opt.train_log,log_file,' ')

#网络参数量计算
def print_network(net):
    num_params = 0
    for param in net.parameters():
            num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

#训练过程记录
def write_log(log_path,log_file,log, refresh=False):
    #print(log)
    log_file.write(log + '\n')
    if refresh:
            log_file.close()
            log_file = open(log_path+'/log.txt', 'a')