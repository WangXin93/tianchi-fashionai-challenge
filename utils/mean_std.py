# 计算图片数据集预处理时的normalization参数


import pandas as pd
import numpy as np
import os
from scipy.misc import imread, imresize
from tqdm import tqdm
import numpy as np

# 最后结果
means = {'coat_length_labels': np.array([ 0.62499291,  0.58789091,  0.57388757]),
         'neckline_design_labels': np.array([ 0.64841022,  0.60368141,  0.58932133]),
         'neck_design_labels': np.array([ 0.63628743,  0.58480037,  0.56696819]),
         'skirt_length_labels': np.array([ 0.63699209,  0.60060385,  0.59294991]),
         'lapel_design_labels': np.array([ 0.63320981,  0.59174452,  0.57950863]),
         'collar_design_labels': np.array([ 0.63914107,  0.59747831,  0.58318485]),
         'pant_length_labels': np.array([ 0.64202558,  0.61319513,  0.60319308]),
         'sleeve_length_labels': np.array([ 0.6470659 ,  0.60789422,  0.59796069])}

stds = {'coat_length_labels': np.array([ 0.30534767,  0.31214217,  0.31014501]),
        'neckline_design_labels': np.array([ 0.30370604,  0.3087055 ,  0.30579874]),
        'neck_design_labels': np.array([ 0.30479367,  0.31424984,  0.31127858]),
        'skirt_length_labels': np.array([ 0.29762986,  0.30413315,  0.30129263]),
        'lapel_design_labels': np.array([ 0.30920743,  0.31700299,  0.31430748]),
        'collar_design_labels': np.array([ 0.30887423,  0.31408858,  0.30994379]),
        'pant_length_labels': np.array([ 0.3016225 ,  0.30119881,  0.29788693]),
        'sleeve_length_labels': np.array([ 0.30389032,  0.30909504,  0.30573339])}


csv_file = '/home/wangx/project/torchfashion/data/fashionAI/{}_train.csv'
root_dir = '/home/wangx/datasets/fashionAI/base'


def get_means_stds(csv_file, root_dir):
    """Return means and stds of each channel

    Args:
        csv_file: (str)
        root_dir: (str)

    Returns:
        mean: 3D array, mean of each channel
        std: 3D array, std of each channel
    """

    df = pd.read_csv(csv_file, names=['image','type','category'])

    means = []
    stds = []
    for i in tqdm(df['image']):
        image = imread(os.path.join(root_dir, i)) / 255.0
        mean = image.mean(axis=(0,1))
        std = image.std(axis=(0,1))
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    mean = means.mean(axis=0)
    # 总方差等于组内均方差+组间方差
    std = stds.mean(axis=0) + stds.std(axis=0)
    return mean, std


def main():
    types = ['skirt_length_labels', 'collar_design_labels', 'neckline_design_labels',
             'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels',
             'lapel_design_labels', 'pant_length_labels']

    means= {}
    stds = {}
    for t in types:
        print('Start get {} ...'.format(t))
        mean, std = get_means_stds(csv_file.format(t), root_dir)
        means[t] = mean
        stds[t] = std
    print("means: {}".format(means))
    print("stds: {}".format(stds))


if __name__ == "__main__":
    main()


