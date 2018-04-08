from utils.predict import predict_model, create_datasets
from utils.metric import mAP, AP, evaluate
import torchvision
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import pandas as pd
import ipdb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Confusion Matrix')
parser.add_argument('--model', type=str, default='resnet18', metavar='M',
                    help='model name')
parser.add_argument('--save_folder', type=str, default='resnet18-zero', metavar='S',
                    help='Subdir of ./log directory to save model.pth files')
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

saved_model = './log/' + args.save_folder + '/{}.pth'
question = './data/fashionAI/{}_test.csv'
root_dir='/home/wangx/datasets/fashionAI/base'

# Create dataloaders for 8 attributes
out = create_datasets(order,
                      csv_file=question,
                      root_dir=root_dir)
dataloaders = out['dataloaders']

labels_alphas = []
results = []

# Iterate each attributes
for t in order:
    if args.model == 'resnet18':
        model_conv = torchvision.models.resnet18()
    elif args.model == 'resnet34':
        model_conv = torchvision.models.resnet34()
    elif args.model == 'resnet50':
        model_conv = torchvision.models.resnet50()

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
    preds = [p.argmax() for p in result]
    df = pd.read_csv(question.format(t),
                     names=['image', 'type', 'category'])
    labels_alpha = df['category']
    ground_truth = [l.index('y') for l in labels_alpha]

    print('confusion matrix ...')
    print(confusion_matrix(ground_truth, preds))
    print('classification report ...')
    print(classification_report(ground_truth, preds))
    print('Ali AP metric score ...')
    print(AP(result, labels_alpha))

    # Save result for mAP
    results.append(result)
    labels_alphas.append(labels_alpha)

print('\nAli mAP metric score ...')
print(mAP(results, labels_alphas))
