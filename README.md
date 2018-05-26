这是**FashionAI全球挑战赛**初赛提交代码。:snake:

队名：**2028部落**
初赛排名：87

# 执行依赖的环境和库
该代码基于PyTorch框架。
```
torch (0.3.1)
torchvision (0.2.0)
pretrainedmodels (0.4.1)
ipdb (0.11)
```


# 文件结构
```
.
├── TRAIN
│   └── run.py            # 训练程序入口
├── TEST
│   ├── flip.py
│   ├── merge_answers.py  # 融合csv文件入口
│   ├── write_answer_a.py # 为每个交叉验证集写答案
│   ├── write_answer_b.py
│   ├── write_answer_c.py
│   ├── write_answer_d.py
│   └── write_answer.py
├── README.md
├── data                  # 存放5折交叉验证的图片索引
│   ├── fashionAI
│   ├── fashionAI_a
│   ├── fashionAI_b
│   ├── fashionAI_c
│   └── fashionAI_d
├── log                   # 存放训练好的模型文件
├── questions
│   ├── question.csv      # 存放测试集文件
│   └── ...
└── utils
    ├── __init__.py
    ├── datasets.py
    ├── mean_std.py
    ├── metric.py
    ├── models.py
    ├── predict.py
    ├── train.py
    └── train_test_csv.py
```


# 训练步骤说明
## 主要训练过程
* 为了预测女装图片的8个属性，分别训练8个卷积神经网络模型预测每个属性。

* 使用5折交叉验证对每个模型训练5次。然后对同一个属性的5个模型的结果做平均得到融合结果。

* 使用测试集数据增强（TTA），具体方法为对原始图片，和其左右反转后的图片做预测。然后取两次平均结果作为最终测试结果。

## 代码使用方法

1. 修改`TRAIN/run.py`文件第59行的`root_dir`为包含训练图片的目录，例如`/home/ubuntu/datasets/base/`，该目录包含`Image`文件夹。

2. 开始训练：
以使用`inceptionresnetv2`模型训练`neck_desing_labels`属性的图片为例，执行：
```
export PYTHONPATH=.
python3 TRAIN/run.py --model inceptionresnetv2 --attribute neck_design_labels --save_folder spam --img_size 299 --epochs 50 --batch_size 24 --csv_folder fashionAI
```
训练结束后的模型文件会被保存为`./log/spam/neck_design_labels.pth`。其中`spam`目录是由`--save_folder`参数定义的，`neck_design_labels.pth`文件名是由训练的`--attribute`参数定义的。通过改变`--csv_folder`参数来选择不同的验证集。

## 主要参数说明
* 预处理与数据增广：
  * 对**训练图片**进行尺寸变换，随机左右镜像反转，随机明度，对比度，饱和度，色度变换和标准化处理。
  * 对**测试图片**进行尺寸编号，标准化处理。
* CNN模型:
  * 对`skirt_length_labels`属性使用[`resnet18`](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)模型。
  * 对其他7个属性使用[`inceptionresnet18`](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py)模型。
  * 在上述的模型的基础上添加一层全连接层`fc`，`fc`层输入维度为1000, 输出维度为服装属性分类数目。
* 训练参数
  * 参数初始化:是用模型在ImageNet数据集上预训练的参数作初始化，对于最后一层全连接层，使用`xavier`初始化
  * 优化器：Adam
  * 学习率：初始学习率为1e-3, 每过7个epoch下降为当前学习率的0.1
  * 训练周期： 50
  * 输入图像尺寸：299x299


# 测试步骤说明
1. 修改`TEST/write_answer.py`, `TEST/write_answer_a.py`, `TEST/write_answer_b.py`, `TEST/write_answer_c.py`, `TEST/write_answer_d.py`这几个文件的第82行`root_dir`为包含待测试图片的目录, 即`Image`的上层目录。
```
python3 TEST/write_answer.py; \
python3 TEST/write_answer_a.py; \
python3 TEST/write_answer_b.py; \
python3 TEST/write_answer_c.py; \
python3 TEST/write_answer_d.py; \
python3 TEST/merge_answers.py --files questions_b/answer.csv questions_b/answer_a.csv questions_b/answer_b.csv questions_b/answer_c.csv questions_d/answer_d.csv --target questions_b/merged1.csv
```

2. 将目录中的图片全部左右反转重新测试一遍。
```
rm questions_b/answer*
python3 TEST/filp.py /path/to/Image/directory
python3 TEST/write_answer.py; \
python3 TEST/write_answer_a.py; \
python3 TEST/write_answer_b.py; \
python3 TEST/write_answer_c.py; \
python3 TEST/write_answer_d.py; \
python3 TEST/merge_answers.py --files questions_b/answer.csv questions_b/answer_a.csv questions_b/answer_b.csv questions_b/answer_c.csv questions_b/answer_d.csv --target questions_b/merged2.csv
```

3. 融合两次的测试结果。
```
python3 TEST/merge_answers.py --files questions_b/merged1.csv questions_b/merged2.csv --target questions_b/merged.csv
```
`merged.csv`即为待提交结果文件。
