import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from .utils import cvtColor, preprocess_input
#from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SiameseDataset(Dataset):
    def __init__(self,train_path,data_path,random):
        self.train_path  = train_path
        self.data_path = data_path
        self.random= random
    def __len__(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return len(lines)

    def __getitem__(self, index):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()  # 应该在循环外部调用一次
        line = lines[index]

        pairs_of_images = [np.zeros((1, 3, 224, 224)) for i in range(2)]
        labels          = np.zeros((1, 1))
        image1_path = self.train_path + "\\" + line.split(',')[0]
        image2_path = self.train_path + "\\" + line.split(',')[1]
        
        image1 = Image.open(image1_path)
        image1   = cvtColor(image1)
        image1 = self.get_random_data(image1, [224,224],  random=self.random)
        image1 = preprocess_input(np.array(image1).astype(np.float32))
        image1 = np.transpose(image1, [2, 0, 1])
        pairs_of_images[0][0, :, :, :] = image1

        image2 =Image.open(image2_path)
        image2   = cvtColor(image2)
        image2 = self.get_random_data(image2, [224,224], random=self.random)
        image2 = preprocess_input(np.array(image2).astype(np.float32))
        image2 = np.transpose(image2, [2, 0, 1])
        pairs_of_images[1][0, :, :, :] = image2

        lable= line.split(',')[2].strip()
        if lable == 'True':
            lable = 1
        else:
            lable = 0
        labels = [lable ]  
        
        return pairs_of_images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #   获得图像的高宽与目标高宽
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            #   将图像多余的部分加上灰条
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data
        
        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #   翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 
        
        # 小范围调整图片亮度、对比度
        if self.rand() < 0.5:
            brightness_factor = 1 + (self.rand() - 0.5) * 0.2  # 亮度变化范围-20%到+20%
            contrast_factor = 1 + (self.rand() - 0.5) * 0.2    # 对比度变化范围-20%到+20%
            image = cv2.convertScaleAbs(np.array(image), alpha=contrast_factor, beta=brightness_factor)
            

        return image

# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return images, labels