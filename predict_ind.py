import numpy as np
from PIL import Image

from siamese import Siamese
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Siamese_mobilenet_v4')
    parser.add_argument('--image1_path', type=str, default="img\Abyssinian_80.png")
    parser.add_argument('--image2_path', type=str, default="img\Abyssinian_81.png")
    parser.add_argument('--model_path', type=str, default='model_data\ind.pth', help='path to weights file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = Siamese(model_path=args.model_path)
    model.net.eval()
    #加载图片
    image_1 = args.image1_path
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error!')

    image_2 = args.image2_path
    try:
        image_2 = Image.open(image_2)
    except:
        print('Image_2 Open Error!')

    #预测
    probability = model.detect_image(image_1,image_2)

    print(probability)

