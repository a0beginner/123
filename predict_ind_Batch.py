import numpy as np
from PIL import Image
import os
from siamese import Siamese
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Siamese_mobilenet_v4')
    parser.add_argument('--file_path', type=str, default=r"L:\个体识别数据集\demo\test.txt")
    parser.add_argument('--test_path', type=str, default=r"L:\个体识别数据集\demo\test")
    parser.add_argument('--model_path', type=str, default='model_data\1.pth', help='path to weights file')
    return parser.parse_args()


args = parse_args()
model = Siamese(model_path=args.model_path)
model.net.eval()
file_path = args.file_path
test_path =  args.test_path
#便利test.txt文件每一行
l=[]
t = []
#计算准确率
count = 0
with open(file_path, 'r') as f:
    lines = f.readlines()
    i = 0
    for line in lines:
        #加载图片
        # img1 = Image.open(test_path + "\\" + line.split(',')[0])
        # img2 = Image.open(test_path + "\\" + line.split(',')[1])
        img1_path = os.path.join(test_path, line.split(',')[0])
        img2_path = os.path.join(test_path, line.split(',')[1])
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        title = line.split(',')[2]

        result = model.detect_image(img1, img2)
        l.append(float(result))
        t.append(title)
        print(i+1,result, title)
        if (result ==True and title == 'True\n') or (result ==False and title == 'False\n'):
            count += 1
        i += 1
print('Accuracy:', count / len(l))

show = False
if show:
    #绘制图表
    import matplotlib.pyplot as plt
    # 绘制所有值的散点图 如果t中的值是True则为绿色，否则为红色
    plt.scatter (range(len(l)), l, c=['g' if i == 'True\n' else 'r' for i in t], marker='o')
    plt.show()
