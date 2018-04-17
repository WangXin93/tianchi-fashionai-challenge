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


def crop(task='neck_design_labels',
         root_dir='/home/ubuntu/datasets/base',
         csv_file='/home/ubuntu/datasets/base/Annotations/label.csv'):
    """
    Args:
        root_dir: Directory with Images folder
        task: Task of images to be cropped
        csv_file: csv file stores all images paths
    """
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
            region.save(path)
        elif img.size[0] > img.size[1]:
            count += 1
            crop = transforms.CenterCrop(img.size[1])
            region = crop(img)
            region.save(path)

    print("Totally {} image are processed".format(count))

if __name__ == "__main__":
    tasks = ['neck_design_labels',
             'collar_design_labels',
             'lapel_design_labels',
             'neckline_design_labels']
    root_dir = os.path.expanduser('~/datasets/fashionAI/base/')
    csv_file = os.path.expanduser('~/datasets/fashionAI/base/Annotations/label.csv')
    for task in tasks:
        print("Start crop {}...".format(task))
        crop(task, root_dir, csv_file)
