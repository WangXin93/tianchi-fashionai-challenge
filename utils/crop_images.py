# 将neck，neckline，collar，lapel图片剪裁保留上半部分
#
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
#import shutil


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def main(root_dir='/home/wangx/datasets/fashionAI/base',
         task='neck_design_labels',
         csv_file='/home/wangx/datasets/fashionAI/base/Annotations/label.csv'):
    """
    Args:
        root_dir: Directory with Images folder
        task: Task of images to be cropped
        csv_file: cav file stores all images paths
    """

    # 裙子任务的目录名
    task = 'neck_design_labels'
    # 热身数据与训练数据的图片标记文件
    #warmup_label_dir = 'data/web/Annotations/skirt_length_labels.csv'
    base_label_dir = '/home/wangx/datasets/fashionAI/base/Annotations/label.csv'

    image_path = []

    #mkdir_if_not_exist(['data/look/data/base/Images', task])
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, tk, label in tokens:
            if tk == task:
                image_path.append((os.path.join(root_dir, path), label))

    #crop to make images'size same
    n = len(image_path)
    count = 0
    for path, label in tqdm(image_path):
        img = Image.open(path)
        if img.size[0] < img.size[1]:
            count += 1
            box = (0,0,img.size[0],img.size[0])
            region = img.crop(box)
            # region.save(path)
        elif img.size[0] > img.size[1]:
            count += 1
            crop = transforms.CenterCrop(img.size[1])
            region = crop(img)
            # region.save(path)

    print("Totally {} image are processed".format(count))

if __name__ == "__main__":
    main()
