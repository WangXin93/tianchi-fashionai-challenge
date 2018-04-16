#
# Demo:
# python3 bin/confusion_matrix.py
#
from utils.predict import predict_model
from utils.datasets import create_dataset
from utils.models import create_model
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Confusion Matrix')
parser.add_argument('--model', type=str, default='resnet18', metavar='M',
                    help='model name')
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
    'collar_design_labels':'inception_v3',
    'lapel_design_labels':'inception_v3',
    'neck_design_labels':'inceptionresnetv2',
    'neckline_design_labels':'inception_v3',
    'pant_length_labels':'inception_v3',
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
    'coat_length_labels':'inceptionresnetv2',
    'collar_design_labels':'inception_v3-zero',
    'lapel_design_labels':'inception_v3-zero',
    'neck_design_labels':'spam',
    'neckline_design_labels':'inception_v3-zero',
    'pant_length_labels':'inception_v3-zero',
    'skirt_length_labels':'resnet18-distort',
    'sleeve_length_labels':'inception_v3-zero',
}

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
               'sleeve_length_labels': ['Sleeveless', 'Cup', 'Short', # no invisible, all visible
                                        'Elbow', '3/4', 'Wrist',
                                        'Long', 'Extra Long',],
              }

saved_model = './log/{}/{}.pth'
# Validation set with ground true labels
question_file = './data/fashionAI/{}_{}.csv'
root_dir='/home/wangx/datasets/fashionAI/base'

labels_alphas = []
results = []
precisions = []

fig, axs = plt.subplots(2, 4, figsize=(21, 12))

# Iterate each attributes
for idx, t in enumerate(order):
    out = create_dataset(t,
                         csv_file=question_file,
                         root_dir=root_dir,
                         phase=['test'],
                         label_mode='alpha',
                         shuffle=False,
                         img_size=ImgSizeKey[t],
                         batch_size=8)
    dataloader = out['dataloaders']['test']

    use_gpu = torch.cuda.is_available()
    model_conv = create_model(model_key=ModelKey[t],
                              pretrained=False,
                              num_of_classes=AttrKey[t],
                              use_gpu=use_gpu)

    ###############################
    # Print classification report #
    ###############################
    print('*'*70)
    print('start analyze {}...'.format(t))
    result = predict_model(model_conv,
                           saved_model.format(SaveFolderKey[t], t),
                           dataloader,
                           use_gpu)
    # Get prediction indices
    preds = [p.argmax() for p in result]

    # Read ground truth labels
    df = pd.read_csv(question_file.format(t, 'test'),
                     names=['image', 'type', 'category'])
    labels_alpha = df['category']
    ground_truth = [l.index('y') for l in labels_alpha]

    print('confusion matrix ...')
    cm = confusion_matrix(ground_truth, preds)
    print(cm)
    print('classification report ...')
    cr = classification_report(ground_truth, preds, target_names=label_names[t], digits=4)
    print(cr)
    precision = cr.splitlines()[-1].split()[3]
    print('Ali AP metric score ...')
    ap = AP(result, labels_alpha)
    print(ap)
    print('*'*70 + '\n')

    # Save result for mAP
    results.append(result)
    labels_alphas.append(labels_alpha)
    precisions.append(float(precision))

    # Draw heatmap
    ax = axs[idx//4, idx%4]
    ax.set_title('{}:{}:{:.4f}'.format(t, precision, ap))
    sns.heatmap(cm,
                ax=ax,
                xticklabels=label_names[t],
                yticklabels=label_names[t],
                annot=True,
                cbar=False,
                fmt='d')

print('Mean basic precision ...')
print(sum(precisions) / len(precisions))

print('Ali mAP metric score ...')
print(mAP(results, labels_alphas))

# Save image of confusion matrix
img_path = Path('./confusion_matrix.png')
print('Image saved in {}'.format(img_path))
plt.savefig(str(img_path))
