from utils.predict import create_dataset, create_datasets, predict_model
import torchvision
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import time
import argparse
from pathlib import Path
import ipdb


order = ['collar_design_labels',
         'neckline_design_labels',
         'skirt_length_labels',
         'sleeve_length_labels',
         'neck_design_labels',
         'coat_length_labels',
         'lapel_design_labels',
         'pant_length_labels']

AttrKey = {
    'coat_length_labels':8,
    'collar_design_labels':5,
    'lapel_design_labels':5,
    'neck_design_labels':5,
    'neckline_design_labels':10,
    'pant_length_labels':6,
    'skirt_length_labels':6,
    'sleeve_length_labels':9,
}


out = create_datasets(order)
dataloaders = out['dataloaders']

results = {}
saved_model = '/home/wangx/project/torchfashion/log/resnet34-transfer/{}.pth'
question = '/home/wangx/datasets/fashionAI/rank/Tests/question_{}.csv'
answer = '/home/wangx/datasets/fashionAI/rank/Tests/answer.csv'

# Iterate each attributes
for t in order:
    model_conv = torchvision.models.resnet34()

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, AttrKey[t])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_conv = model_conv.cuda()

    print('start write {}...'.format(t))
    dataloader = dataloaders[t]
    result = predict_model(model_conv,
                            saved_model.format(t),
                            dataloader,
                            use_gpu)
    results[t] = result

    # Read lines of question files
    lines = open(question.format(t)).readlines()
    assert len(lines) == len(result)
    # Write to answer.csv
    with open(answer, 'a') as ansf:
        for line, probs in zip(lines, result):
            # Change ? mark to probabilities
            line = line.replace('?', ';'.join('{:.4f}'.format(i) for i in probs))
            ansf.write(line)
