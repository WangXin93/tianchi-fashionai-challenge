#
# Demo:
# python3 bin/transfer_learning.py --epochs 3 --pretrained True --save_folder spam --attribute neck_design_labels
# 
from utils.datasets import create_dataset
from utils.train import train_model, train_model_noval
from utils.models import create_model
import torchvision
from torch import nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import time
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description='FashionAI')
parser.add_argument('--model', type=str, default='resnet18', metavar='M',
                    help='model name')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='Chosed optimizer')
parser.add_argument('--attribute', type=str, default='neck_design_labels', metavar='A',
                    help='fashion attribute (default: coat_length_labels)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--save_folder', type=str, default='spam', metavar='S',
                    help='Subdir of ./log directory to save model.pth files')
parser.add_argument('--pretrained', type=str, default='False', metavar='P', 
                    choices=['True', 'False'],
                    help='If True, only train last layer of model')
parser.add_argument('--trained_model', type=str, default=None, metavar='M', 
                    help='File name of a trained model before.')
parser.add_argument('--img_size', type=int, default=224, metavar='S',
                    help='Size of input images.')
parser.add_argument('--batch_size', type=int, default=32, metavar='B',
                    help='Batch number of input images')
parser.add_argument('--verbose', action='store_true',
                    help='If use verbose flag, more detailed training will be printed')

args = parser.parse_args()


AttrKey = {
    'coat_length_labels':8,
    'collar_design_labels':5,
    'lapel_design_labels':5,
    'neck_design_labels':5,
    'neckline_design_labels':10,
    'pant_length_labels':6,
    'skirt_length_labels':6,
    'sleeve_length_labels':9, }

# Create dataloader
csv_file = './data/fashionAI_b/{}_{}.csv'
out = create_dataset(args.attribute,
                     csv_file=csv_file,
                     img_size=args.img_size, 
                     batch_size=args.batch_size)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

# Create CNN model
use_gpu = torch.cuda.is_available()
model_conv = create_model(model_key=args.model,
                          pretrained=eval(args.pretrained),
                          num_of_classes=AttrKey[args.attribute],
                          use_gpu=use_gpu,)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
if eval(args.pretrained):
    parameters_totrain = model_conv.fc.parameters()
else:
    parameters_totrain = model_conv.parameters()

# Choose optimizer for training
if args.optimizer == 'SGD':
    optimizer_conv = optim.SGD(parameters_totrain, lr=0.001, momentum=0.9)
elif args.optimizer == 'Adam':
    optimizer_conv = optim.Adam(parameters_totrain, lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# Make save folder
save_folder = Path('log') / Path(args.save_folder)
if not save_folder.exists():
    save_folder.mkdir(parents=True)

save_file = str(save_folder / Path(args.attribute+'.pth'))

# Load trained model before if define it
if args.trained_model:
    print("Load trained model from {}".format(args.trained_model))
    model_conv.load_state_dict(torch.load(args.trained_model))
# Kick off the train
model_conv = train_model(model_conv,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         dataloaders,
                         dataset_sizes,
                         use_gpu,
                         save_file,
                         num_epochs=args.epochs,
                         verbose=args.verbose)

