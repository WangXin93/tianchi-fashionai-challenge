from utils.datasets import create_dataset
from utils.train import train_model
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
parser.add_argument('--model', type=str, default='resnet34', metavar='M',
                    help='model name')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='Chosed optimizer')
parser.add_argument('--attribute', type=str, default='coat_length_labels', metavar='A',
                    help='fashion attribute (default: coat_length_labels)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--save_folder', type=str, default='resnet34', metavar='S',
                    help='Subdir of ./log directory to save model.pth files')
parser.add_argument('--pretrained', type=str, default='True', metavar='P', 
                    choices=['True', 'False'],
                    help='Set this to False to use un-pretrained model')
args = parser.parse_args()


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


datasetd = create_dataset(args.attribute)
dataloaders = datasetd['dataloaders']
dataset_sizes = datasetd['dataset_sizes']

if args.model == 'resnet18':
    model_conv = torchvision.models.resnet18(pretrained=True)
elif args.model == 'resnet34':
    model_conv = torchvision.models.resnet34(pretrained=True)
elif args.model == 'resnet50':
    model_conv = torchvision.models.resnet50(pretrained=True)
elif args.model == 'resnet101':
    model_conv = torchvision.models.resnet101(pretrained=True)

if eval(args.pretrained):
    # Lock parameters for transfer learning
    for param in model_conv.parameters():
        param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, AttrKey[args.attribute])

# Initialize newly added module parameters
nn.init.xavier_uniform(model_conv.fc.weight)
nn.init.constant(model_conv.fc.bias, 0)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model_conv = model_conv.cuda()

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
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         dataloaders,
                         dataset_sizes,
                         use_gpu,
                         num_epochs=args.epochs)

save_folder = Path('log') / Path(args.save_folder)
if not save_folder.exists():
    save_folder.mkdir(parents=True)

# Save model
torch.save(model_conv.state_dict(), str(save_folder / Path(args.attribute+'.pth')))
print('Saved to {}'.format(str(save_folder / Path(args.attribute+'.pth'))))
