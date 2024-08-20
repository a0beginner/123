from ultralytics import YOLO
import argparse
# 预测参数官方详解链接：https://docs.ultralytics.com/modes/predict/
def parse_args():
    parser = argparse.ArgumentParser(description='Predict  model')
    parser.add_argument('--model_path', type=str, default='model_data\cls.pt', help='dataset.yaml path')
    parser.add_argument('--image_path', type=str, default='img', help='dataset split')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    model=YOLO(args.model_path)
    source = args.image_path
    #source = r"E:\huaweibei\dogs_cats_datasets\datasets\images8_2\test\Siamese\Siamese_222.png"
    results = model(source)  # predict on an image
    imgsz = 224
# Process re sults list
    for result in results:
        probs = result.probs  # Probs object for classification outputs
        print("图片地址：",result.path,"类别编号：",probs.top1,"类别名称：",result.names[probs.top1])
        # print(probs.top1)  # print top class name and confidence
        # print(result.names[probs.top1])
        #result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk