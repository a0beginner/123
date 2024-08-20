from ultralytics import YOLO
import argparse
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/
def parse_args():
    parser = argparse.ArgumentParser(description='Train  model')
    parser.add_argument('--data',  default='')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()

    model = YOLO('ultralytics/cfg/models/v8/yolov8m-cls-faster-CGLU.yaml')

    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=args.data,
                cache=False,
                imgsz=224,
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                #amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )