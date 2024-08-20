# 基于YOLOv8和孪生网络（Siamese Network）的猫/狗细粒度识别代码说明文档

---

本代码库实现了两个主要功能：基于YOLOv8的种类识别和基于孪生神经网络（Siamese Network）的个体识别。YOLOv8负责识别图片中猫和狗的种类，而孪生网络则用于评估两张图片（猫/狗）的相似性。项目中使用的主干特征提取网络为MobileNet V4。

<iframe height=498 width=900 src="https://veed.io/view/417e8a95-3356-4b26-9efe-a7b4dec23351"></iframe>

## 目录

1. [所需环境](#所需环境)
2. [训练数据集](#训练数据集)
3. [预测步骤](#预测步骤)
4. [训练步骤](#训练步骤)
5. [参考资料](#参考资料)

### 所需环境

- 创建虚拟环境
   ```bash
  conda create -n {name} python=x.x
  ```
- 激活环境
   ```bash
  conda activate {name}
  ```
- 安装[pytorch](#https://pytorch.org/get-started/previous-versions/)
   ```bash
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
- 安装其他需要的包
   ```bash
  pip install ultralytics==8.2.63 einops==0.8.0 timm==1.0.7 tensorboard==2.16.2 
  ```

### 训练数据集

种类识别数据集下载地址：
- 链接：[点击此处](https://pan.baidu.com/s/1WKlhHB0g0wUDJntx5bMpmw?pwd=ki6m )
- 提取码：[ki6m]

个体识别数据集下载地址：
- 链接：[点击此处](https://pan.baidu.com/s/1hdAHk8rwbzY9LeFaMK7oPg?pwd=w0ph )
- 提取码：[w0ph ]

我们提供了两个训练好的权重文件用于品种以及个体识别任务存放在\model_data目录下：
- `cls.pth`：对应于种类识别训练好的权重。
- `ind.pth`：对应于个体识别训练好的权重。

### 预测步骤

#### a. 种类识别

1. 利用训练好的模型进行预测（无标签）

   使用predict_cls.py进行单/多张图片预测，需传入的参数有 --model_path（模型权重地址） --image_path（图片地址/图片所在文件夹）。

   ```bash
   python predict_cls.py  --model_path /path/to/your/model_weights.pt --image_path  path/to/image.jpg
   ```

2. 验证模型效果（图片所在文件夹名为类别标签）

   使用val_cls.py进行验证，需传入的参数有 --model_path（模型权重地址） --test_path（测试图片文件夹地址，需按后文  [训练步骤a](###a. 训练自己的种类识别模型)中格式存放）。

   ```bash
   python val_cls.py  --model_path /path/to/your/model_weights.pt  --test_path  path/to/test
   ```

#### b. 使用两张图像评估个体识别模型（无标签）

运行 `predict_ind.py`，依次输入两个图片的位置：
```bash
python predict_ind.py  --model_path /path/to/your/model_weights.pth --image1_path path/to/image1.jpg --image2_path path/to/image2.jpg
```
示例：
```bash
python predict_ind.py --model_path D:\Siamese\logs\best_epoch_weights.pth  --image1_path D:\img\test\1.png --image2_path D:\img\test\2.png  
```

#### c. 批量图像评估个体识别模型（带标签）

运行 `predict_ind_Batch.py`进行批量预测： 所需标签文件格式见后文  [训练步骤b.1.(2)](###b. 训练自己的相似性比较模型)
```bash
python predict_ind_Batch.py --model_path /path/to/your/model_weights.pth  --file_path path/to/test.txt --test_path path/to/test 
```
示例：
```bash
python predict_ind_Batch.py  --model_path D:\Siamese\logs\best_epoch_weights.pth  --file_path D:\Siamese\img\test.txt --test_path D:\Siamese\img\test 
```

#### d.图形化界面

为便于操作我们还提供了一个图形化窗口ui.py便于测试，如需使用图形化界面需安装tkinter库 

```bash
pip install tkinter
```

1.模型加载

<img src=".\pic\1 (1).png" alt="1 (1)" style="zoom:67%;" />

如下图时，则为加载成功

<img src=".\pic\1 (5).png" alt="1 (5)" style="zoom:67%;" />

2.个体识别

1）一组个体（两只个体）识别,预测结果如图

<img src=".\pic\1 (4).png" alt="1 (4)" style="zoom:67%;" />



2）批量个体识别步骤及预测结果

<img src=".\pic\1 (3).png" alt="1 (3)" style="zoom:67%;" />

3.种类识别

1）单个图片的种类识别

<img src=".\pic\1 (6).png" alt="1 (6)" style="zoom:67%;" />

2）批量的种类识别

<img src=".\pic\1 (2).png" alt="1 (2)" style="zoom:67%;" />

### 训练步骤

#### a. 训练自己的种类识别模型

1.将数据按以下格式组织：

    images-/
        |-- train/
        |   |-- 品种1/
        |   |   |-- 10008_airplane.png
        |   |   |-- 10009_airplane.png
        |   |   |-- ...
        |   |
        |   |-- 品种2/
        |   |   |-- 1000_automobile.png
        |   |   |-- 1001_automobile.png
        |   |   |-- ...
        |   |-- ...
        |
        |-- test/
        |   |-- 品种1/
        |   |   |-- 10_airplane.png
        |   |   |-- 11_airplane.png
        |   |   |-- ...
        |   |-- ...
        |
        |-- val/ (optional)
        |   |-- 品种1/
        |   |   |-- 105_airplane.png
        |   |   |-- 106_airplane.png
        |   |   |-- ...
        |   |-- ...


2.训练步骤：

运行 `train_cls.py` 开始训练：：

```bash
python train_cls.py  --data path/to/image
```

#### b. 训练自己的相似性比较模型

1.将数据集按照以下格式进行组织：

（1）图片存放样式:

   ```plaintext
ImgsUnit-/
    |-- train/
    |   |-- img1.png
    |   |-- img2.png
    |   |-- ...
    |-- test/
    |   |-- img1.png
    |   |-- img2.png
    |   |-- ...
    |-- val/
    |   |-- img1.png
    |   |-- img2.png
    |   |-- ...
   ```

（2）标签文件 `lable.txt`：

   ```bash
   img1.png,img2.png,True
   img3.png,img4.png,False
   ……
   ```

2.训练步骤：

（1）按上述格式放置数据集和对应标签文件。

（2）运行 `train_ind.py` 开始训练：

```bash
python train_ind.py --train_path img/train --train_txt_path img/train.txt --val_path img/val --val_txt_path img/val.txt
```

更多参数设置请参考 `train_ind.py` 中17~45行代码

为方便生成训练数据，我们提供了相应的数据集及标签生成代码，存放在 `others` 目录下。可将数据按训练步骤 (a.1)中格式存放(每个个体图片存放在一个文件夹中)，使用代码 `train_ind.py`，生成个体识别训练数据集。



### 参考资料

- [YOLOv8 Documentation](https://docs.ultralytics.com/zh/models/yolov8/)
- [Siamese Network Tutorial](https://github.com/bubbliiiing/Siamese-pytorch)

