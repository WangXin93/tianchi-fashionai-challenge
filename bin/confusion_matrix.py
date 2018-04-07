from utils.predict import predict_model
from utils.dataset import create_dataset
from utils.metric import mAP, AP, evaluate
import torchvision
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import ipdb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Confusion Matrix')
parser.add_argument('--model', type=str, default='resnet34', metavar='M',
                    help='model name')
parser.add_argument('--save_folder', type=str, default='resnet34-transfer', metavar='S',
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


results = {}
labels_dict = {}

saved_model = './log/' + args.save_folder + '/{}.pth'
val_file = './data/fashionAI/{}_test.csv'

# Iterate each attributes
for t in order:
    if args.model == 'resnet18':
        model_conv = torchvision.models.resnet18()
    elif args.model == 'resnet34':
        model_conv = torchvision.models.resnet34()

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, AttrKey[t])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_conv = model_conv.cuda()

    print('start analyze {}...'.format(t))
    # Create dataloaders for 8 attributes
    out = create_dataset(t)
    dataloader = out['dataloader']

    result = predict_model(model_conv,
                           saved_model.format(t),
                           dataloader,
                           use_gpu)
    results[t] = result

    # Get prediction indices
    preds = [p.index(max(p)) for p in resullt]
    # Get labels' indices of 'y'
    df = pd.read_csv(val_file.format(t), names=['image','type','category'])
    labels_dict[t] = df['category']
    labels = [l.index('y') for l in df['category']]

    print('confusion matrix ...')
    print(confusion_matrix(labels, preds))
    print('classification report ...')
    print(classification_report(labels, preds))
    print('Ali AP metric score ...')
    print(AP(result, df['category']))

#print('Ali mAP metric score ...')
#print(mAP(results.values(), labels_dict.values()))
