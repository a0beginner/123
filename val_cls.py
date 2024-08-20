from ultralytics import YOLO
import yaml
import os
import argparse
# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#key-features-of-val-mode
def parse_args():
    parser = argparse.ArgumentParser(description='val model')
    parser.add_argument('--model_path', type=str, default='model_data\yolov8m_faster.pt', help='dataset.yaml path')
    parser.add_argument('--test_path', type=str, default='val', help='dataset split')
    return parser.parse_args()
def updata_yaml(k,v):
    old_data=yaml.load(open("ultralytics\cfg\datasets\my.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    old_data[k]=v #修改读取的数据（k存在就修改对应值，k不存在就新增一组键值对）
    with open("ultralytics\cfg\datasets\my.yaml", "w", encoding="utf-8") as f:
        yaml.dump(old_data,f)

if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.model_path)
    updata_yaml("test",args.test_path)
    model.val(data=r'my.yaml',
              split='test',
              imgsz=224,
              batch=16,
              # iou=0.7,
              # rect=False,
              project='runs/val',
              name='exp',
              )