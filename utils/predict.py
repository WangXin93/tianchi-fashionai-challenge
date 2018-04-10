# 预测结果并写入question.csv文件

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from utils.mean_std import means, stds
import time
from torch.autograd import Variable


csv_file = '/home/wangx/datasets/fashionAI/rank/Tests/question.csv'
order = ['collar_design_labels',
         'neckline_design_labels',
         'skirt_length_labels',
         'sleeve_length_labels',
         'neck_design_labels',
         'coat_length_labels',
         'lapel_design_labels',
         'pant_length_labels']


def create_question_csv(csv_file=csv_file):
    """Divide question.csv into 8 part in terms of type, and save 
    them in the same directory as csv_file.

    Args:
        csv_file (str): Path of question.csv file.

    """
    df = pd.read_csv(csv_file, names=['image', 'type', 'answer'])
    for t in df['type'].unique():
        tdf = df[df['type']==t]
        fname = csv_file.replace('question', '{}_test'.format(t))
        tdf.to_csv(fname, index=False, header=False)


def predict_model(model, saved_model, dataloader, use_gpu):
    """Predict probabilities based on trained model

    Args:
        model: Defined model from torchvision
        saved_model (str): Path of trained model parameters
        dataloader: Dataloader of test images
        use_gpu: Where to use gpu

    Returns:
        result (list): Results of each sample

    """
    result = []

    since = time.time()

    model.train(False)  # Set model to evaluate mode
    # Load parameters
    model.load_state_dict(torch.load(saved_model))

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs = data['image']

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs)
        
        # Convert to cpu
        probs = probs.data.cpu().numpy()

        for p in probs:
            result.append(p)

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return result
