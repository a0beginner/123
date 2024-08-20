import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenetv4 import mobilenetv4_conv_small




def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.net = mobilenetv4_conv_small()

        
        self.fully_connect1 = torch.nn.Linear(11520, 512)#200704  //11520 //8640
        #添加dropout层
        self.dropout = torch.nn.Dropout(0.5)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        #   将两个输入传入到主干特征提取网络
        #print("x1shape",x1.shape)
        x1 = self.net.conv0(x1)
        x1 = self.net.layer1(x1)
        x1 = self.net.layer2(x1)
        x1 = self.net.layer3(x1)
        x1 = self.net.layer4(x1)
        x1 = self.net.layer5(x1)
        x2 = self.net.conv0(x2)
        x2 = self.net.layer1(x2)
        x2 = self.net.layer2(x2)
        x2 = self.net.layer3(x2)
        x2 = self.net.layer4(x2)
        x2 = self.net.layer5(x2)
        x1 = self.net.avgpool(x1)
        x2 = self.net.avgpool(x2)

        #   相减取绝对值，取l1距离   
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # #   进行两次全连接
        x = self.fully_connect1(x)
        x = self.dropout(x)
        x = self.fully_connect2(x)
        return x