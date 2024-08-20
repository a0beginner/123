import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.siamese_mobilenet_v4 import Siamese
from utils.callbacks import LossHistory
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch
import argparse

def get_args():
    params = argparse.ArgumentParser(description='Siamese Training')
    params.add_argument('--Cuda', type=bool, default=True, help='是否使用Cuda')
    params.add_argument('--distributed', type=bool, default=False, help='是否使用单机多卡分布式运行')
    params.add_argument('--sync_bn', type=bool, default=False, help='是否使用sync_bn，DDP模式多卡可用')
    params.add_argument('--fp16', type=bool, default=True, help='是否使用混合精度训练')
    params.add_argument('--input_shape', type=list, default=[224, 224], help='输入图像的大小')
    params.add_argument('--pretrained', type=bool, default=False, help='pretrained')
    params.add_argument('--ngpus_per_node', type=int, default=torch.cuda.device_count(), help='ngpus per node')
    params.add_argument('--model_path', type=str, default='')#
    params.add_argument('--Init_Epoch', type=int, default=0, help='init epoch')
    params.add_argument('--Epoch', type=int, default=800, help='epoch')
    params.add_argument('--batch_size', type=int, default=32, help='batch size')
    params.add_argument('--Init_lr', type=float, default=1e-2, help='init lr')
    params.add_argument('--optimizer_type', type=str, default='sgd', help='optimizer type:adam、sgd')
    params.add_argument('--momentum', type=float, default=0.9, help='momentum')
    params.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    params.add_argument('--lr_decay_type', type=str, default='cos', help='lr decay type')
    params.add_argument('--save_period', type=int, default=5, help='save period')
    params.add_argument('--save_dir', type=str, default='logs', help='训练日志保存路径')
    params.add_argument('--num_workers', type=int, default=8, help='num workers')
    params.add_argument('--num_train', type=int, default=0, help='num train')
    params.add_argument('--num_val', type=int, default=0, help='num val')
    params.add_argument('--train_path', type=str, default=r'L:\个体识别数据集\demo\train', help='训练图片文件夹路径')
    params.add_argument('--train_txt_path', type=str, default=r'L:\个体识别数据集\demo\train.txt', help='训练图片标签txt路径')
    params.add_argument('--val_path', type=str, default=r'L:\个体识别数据集\demo\val', help='验证图片文件夹路径')
    params.add_argument('--val_txt_path', type=str, default=r'L:\个体识别数据集\demo\val.txt', help='验证图片标签txt路径')
    return params.parse_args()




if __name__ == "__main__":
    args = get_args()
    Cuda = args.Cuda
    distributed = args.distributed
    sync_bn = args.sync_bn
    fp16 = args.fp16
    input_shape = args.input_shape
    pretrained = args.pretrained
    ngpus_per_node = args.ngpus_per_node
    model_path = args.model_path
    Init_Epoch = args.Init_Epoch
    Epoch = args.Epoch
    batch_size = args.batch_size
    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    optimizer_type = args.optimizer_type
    momentum = args.momentum
    weight_decay = args.weight_decay
    lr_decay_type = args.lr_decay_type
    save_period = args.save_period
    save_dir = args.save_dir
    num_workers = args.num_workers
    num_train = args.num_train
    num_val = args.num_val
    train_path = args.train_path
    train_txt_path = args.train_txt_path
    val_path = args.val_path
    val_txt_path = args.val_txt_path

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        local_rank      = 0
        rank            = 0


    model = Siamese(input_shape, pretrained)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #   根据预训练权重的Key和模型的Key进行加载
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        #   显示没有匹配上的Key
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    loss = nn.BCEWithLogitsLoss()

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #   多卡同步Bn
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    if Cuda:
        if distributed:
            #   多卡平行运行
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()


    train_ratio = 0.9
  
    with open(train_txt_path, 'r') as f:
        train_lines = f.readlines()
    num_train   = len(train_lines)
    with open(val_txt_path, 'r') as f:
        val_lines = f.readlines()
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step  = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    if True:
        #   判断当前batch_size，自适应调整学习率
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #   根据optimizer_type选择优化器
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        #   判断每一个世代的长度
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        train_dataset   = SiameseDataset(train_path, train_txt_path, random=True)
        val_dataset     = SiameseDataset( val_path,   val_txt_path , random=False)
        
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
        if local_rank == 0:
            loss_history.writer.close()
