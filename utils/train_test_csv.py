# 将warm的图片全部复制到base数据集
# 将warm中的label文件添加到label.csv后面
# 将label.csv按属性分为8个文件，每个文件做训练集，测试集分割

import pandas as pd
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv('/home/wangx/datasets/fashionAI/base/Annotations/label.csv',
                 names=['image','type','category'])

# 按type分成8个DataFrame
dfs = dict()
for t in df['type'].unique():
    dfs[t] = df[df['type'] == t]
    # 进行训练，测试分割
    X_train, X_test = train_test_split(dfs[t],
                                       test_size=0.2,
                                       random_state=666)
    # 分别存储训练集，测试集文件
    train_fname = os.path.join('/home/wangx/project/torchfashion/data/fashionAI',
                               t) + '_train.csv'
    test_fname = os.path.join('/home/wangx/project/torchfashion/data/fashionAI',
                               t) + '_test.csv'
    X_train.to_csv(train_fname, index=False, header=False)
    X_test.to_csv(test_fname, index=False, header=False)




