#
# Demo:
# python3 bin/write_answer.py
#

from utils.predict import predict_model
from utils.datasets import create_dataset
from utils.models import create_model
import torchvision
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import time
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='Write Answer')
parser.add_argument('--answer', type=str, default='answer', metavar='A',
                    help='File name of answers relative to ./questions/ directory')
args = parser.parse_args()


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

ModelKey = {
    'coat_length_labels':'inceptionresnetv2',
    'collar_design_labels':'inceptionresnetv2',
    'lapel_design_labels':'inceptionresnetv2',
    'neck_design_labels':'inceptionresnetv2',
    'neckline_design_labels':'inception_v3',
    'pant_length_labels':'inceptionresnetv2',
    'skirt_length_labels':'resnet18',
    'sleeve_length_labels':'inception_v3',
}

ImgSizeKey = {
    'coat_length_labels':299,
    'collar_design_labels':299,
    'lapel_design_labels':299,
    'neck_design_labels':299,
    'neckline_design_labels':299,
    'pant_length_labels':299,
    'skirt_length_labels':224,
    'sleeve_length_labels':299,
}

SaveFolderKey = {
    'coat_length_labels':'inceptionresnetv2-aws',
    'collar_design_labels':'inceptionresnetv2',
    'lapel_design_labels':'inceptionresnetv2',
    'neck_design_labels':'inceptionresnetv2-aws',
    'neckline_design_labels':'inception_v3-zero',
    'pant_length_labels':'inceptionresnetv2',
    'skirt_length_labels':'resnet18-distort',
    'sleeve_length_labels':'inception_v3-zero',
}


saved_model = './log/{}/{}.pth'
question_file = './questions/{}_{}.csv'
root_dir = '/home/wangx/datasets/fashionAI/rank'
answer = './questions/' + args.answer + '.csv'

# Iterate each attributes
for t in order:
    # Create dataloader for each attribute
    out = create_dataset(t,
                         csv_file=question_file,
                         root_dir=root_dir,
                         phase=['test'],
                         label_mode='?',
                         shuffle=False,
                         img_size=ImgSizeKey[t],
                         batch_size=8)
    dataloader = out['dataloaders']['test']

    # Create CNN model
    use_gpu = torch.cuda.is_available()
    model_conv = create_model(model_key=ModelKey[t],
                              pretrained=False,
                              num_of_classes=AttrKey[t],
                              use_gpu=use_gpu)

    print('start write {}...'.format(t))
    result = predict_model(model_conv,
                           saved_model.format(SaveFolderKey[t], t),
                           dataloader,
                           use_gpu)

    # Read lines of question files
    lines = open(question_file.format(t, 'test')).readlines()
    assert len(lines) == len(result)

    # Write to answer.csv
    with open(answer, 'a') as f:
        for line, probs in zip(lines, result):
            # Change ? mark to probabilities
            line = line.replace('?', ';'.join('{:.4f}'.format(i) for i in probs))
            f.write(line)
