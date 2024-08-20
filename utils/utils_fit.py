import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((label) * torch.pow(torch.clamp(abs(euclidean_distance)-1, min=0.0), 2) +     # calmp夹断用法
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive
    
loss_func = ContrastiveLoss(margin=3)
def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    #print("model_train",model_train)
    for iteration, batch in  enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                targets = targets.unsqueeze(1) 

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            output  = loss_func(outputs, targets)

            output.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                output = loss_func(outputs, targets)
                #print("loss_func",loss_func(outputs, targets))
                # print("output",output)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(output).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            # 将输出小于1的标签设置为1，输出大于1的标签设置为0
            labels = torch.where(outputs < 2, torch.ones_like(outputs), torch.zeros_like(outputs))
            # 计算准确率
            accuracy = torch.mean((labels == targets).float())
            # equal       = torch.eq(torch.round(nn.Sigmoid()(outputs[2])), targets)
            # accuracy    = torch.mean(equal.float())

        total_loss      += output.item()
        total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in  enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                targets = targets.unsqueeze(1) 
                
            optimizer.zero_grad()
            outputs = model_train(images)
            output  = loss_func(outputs, targets)#loss_func(outputs, targets)
            # 将输出小于1的标签设置为1，输出大于1的标签设置为0
            labels = torch.where(outputs < 2, torch.ones_like(outputs), torch.zeros_like(outputs))
            
            # 计算准确率
            accuracy = torch.mean((labels == targets).float())
            #equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            #accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))