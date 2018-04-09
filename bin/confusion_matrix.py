from utils.predict import predict_model, create_datasets
from utils.metric import mAP, AP, evaluate
import torchvision
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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

#label_names = {'skirt_length_labels': ['Invisible', 'Short Length', 'Knee Length', 'Midi Length',
#                                       'Ankle Length', 'Floor Length',],
#               'coat_length_labels': ['Invisible', 'High Waist Length', 'Regular Length',
#                                      'Long Length', 'Micro Length', 'Knee Length', 'Midi Length',
#                                      'Ankle&Floor Length',],
#               'collar_design_labels': ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar',
#                                        'Rib Collar',],
#               'lapel_design_labels': ['Invisible', 'Notched', 'Collarless', 'Shawl Collar', 
#                                       'Plus Size Shawl',],
#               'neck_design_labels': ['Invisible', 'Turtle Neck', 'Ruffle Semi-High Collar', 
#                                      'Low Turtle Neck', 'Draped Collar',],
#               'neckline_design_labels': ['Invisible', 'Strapless Neck', 'Deep V Neckline',
#                                          'Straight Neck', 'V Neckline', 'Square Neckline',
#                                          'Off Shoulder', 'Round Neckline', 'Sweat Heart Neck',
#                                          'One Shoulder Neckline',] ,
#               'pant_length_labels': ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length',
#                                      'Cropped Pant','Full Length',],
#               'sleeve_length_labels': ['Invisible', 'Sleeveless', 'Cup Sleeves', 'Short Sleeves',
#                                        'Elbow Sleeves', '3/4 Sleeves', 'Wrist Length',
#                                        'Long Sleeves', 'Extra Long Sleeves',],
#              }
# Make it shorter
label_names = {'skirt_length_labels': ['Invisible', 'Short', 'Knee', 'Midi',
                                       'Ankle', 'Floor',],
               'coat_length_labels': ['Invisible', 'High Waist', 'Regular',
                                      'Long', 'Micro', 'Knee', 'Midi',
                                      'Ankle&Floor',],
               'collar_design_labels': ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan',
                                        'Rib Collar',],
               'lapel_design_labels': ['Invisible', 'Notched', 'Collarless', 'Shawl ', 
                                       'Plus Size',],
               'neck_design_labels': ['Invisible', 'Turtle', 'Ruffle', 
                                      'Low Turtle', 'Draped',],
               'neckline_design_labels': ['Invisible', 'Strapless', 'Deep V',
                                          'Straight', 'V', 'Square',
                                          'Off Shoulder', 'Round', 'Sweat Heart Neck',
                                          'One Shoulder',] ,
               'pant_length_labels': ['Invisible', 'Short Pant', 'Mid', '3/4',
                                      'Cropped Pant','Full',],
               'sleeve_length_labels': ['Sleeveless', 'Cup', 'Short', # all visible
                                        'Elbow', '3/4', 'Wrist',
                                        'Long', 'Extra Long',],
              }

saved_model = './log/' + args.save_folder + '/{}.pth'
question = './data/fashionAI/{}_test.csv'
root_dir='/home/wangx/datasets/fashionAI/base'

# Create dataloaders for 8 attributes
out = create_datasets(order,
                      csv_file=question, root_dir=root_dir)
dataloaders = out['dataloaders']

labels_alphas = []
results = []

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# Iterate each attributes
for idx, t in enumerate(order):
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

    print('*'*70)
    print('start analyze {}...'.format(t))
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
    cm = confusion_matrix(ground_truth, preds)
    print(cm)
    print('classification report ...')
    print(classification_report(ground_truth, preds, target_names=label_names[t]))
    print('Ali AP metric score ...')
    print(AP(result, labels_alpha))
    print('*'*70 + '\n')

    # Save result for mAP
    results.append(result)
    labels_alphas.append(labels_alpha)

    # Draw heatmap
    # plt.subplot(idx//4, idx%4)
    ax = axs[idx//4, idx%4]
    ax.set_title(t)
    sns.heatmap(cm,
                ax=ax,
                xticklabels=label_names[t],
                yticklabels=label_names[t],
                annot=True,
                cbar=False,
                fmt='d')

print('Ali mAP metric score ...')
print(mAP(results, labels_alphas))

img_path = Path('log') / Path(args.save_folder) / Path('confusion_matrix.png')
plt.savefig(str(img_path))
