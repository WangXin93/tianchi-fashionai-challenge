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

datasetd = create_dataset('coat_length_labels')
dataloaders = datasetd['dataloaders']
dataset_sizes = datasetd['dataset_sizes']

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, AttrKey['coat_length_labels'])

use_gpu = torch.cuda.is_available()
if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv,
                         criterion,
                         optimizer_conv,
                         exp_lr_scheduler,
                         dataloaders,
                         dataset_sizes,
                         num_epochs=100)

