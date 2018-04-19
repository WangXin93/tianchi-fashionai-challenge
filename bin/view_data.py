import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
# import Augmentor


class FashionAttrsDataset(Dataset):
    """Fashion Attributes dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fashion_frame = pd.read_csv(csv_file, names=['image', 'type','category'])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fashion_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.fashion_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        image = Image.open(img_name).convert('RGB')
        category = self.fashion_frame.iloc[idx, 2]
        # category = category[0] != ('y')
        category = category.index('y')

        if self.transform:
            transformed_image = self.transform(image)
        image = transforms.Resize((224,224))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.63699209,  0.60060385,  0.59294991], [ 0.29762986,  0.30413315,  0.30129263])(image)
        sample = {'image': image, 'category': category, 'transformed_image': transformed_image}
        return sample


#p = Augmentor.Pipeline()
#p.rotate(probability=1.0, max_left_rotation=8, max_right_rotation=8)
#p.random_distortion(probability=1.0, grid_width=4, grid_height=4, magnitude=10)

data_transforms = {
    'train': transforms.Compose([
#        p.torch_transform(),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ColorJitter(brightness=32./255.,
                                           contrast=0.5,
                                           saturation=0.5,
                                           hue=0.2),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.4),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.63699209,  0.60060385,  0.59294991], [ 0.29762986,  0.30413315,  0.30129263])

    ]),
    'val': transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.63699209,  0.60060385,  0.59294991], [ 0.29762986,  0.30413315,  0.30129263])
    ]),
}

csv_file = './data/fashionAI/neck_design_labels_train.csv'
root_dir = '/home/wangx/datasets/fashionAI/base/'

use_gpu = torch.cuda.is_available()

image_datasets = {x: FashionAttrsDataset(csv_file,
                                         root_dir,
                                         data_transforms[x])
                      for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=16,
                                 shuffle=True,
                                 num_workers=4)
                   for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def imshow(inp, ax, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.63699209,  0.60060385,  0.59294991])
    std = np.array([0.29762986,  0.30413315,  0.30129263])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    #plt.pause(1)  # pause a bit so that plots are updated


# Get a batch of training data
next_batch = next(iter(dataloaders['train']))
inputs, o_images, classes = next_batch['transformed_image'], next_batch['image'], next_batch['category']
classes = [str(i) for i in classes]

# Make a grid from batch
fig, axs = plt.subplots(2, 1, figsize=(16, 9))
axs[0].text(0, -5, ','.join(classes), fontsize=24)
out = torchvision.utils.make_grid(inputs)
imshow(out, axs[0], title='transformed images')

out = torchvision.utils.make_grid(o_images)
imshow(out, axs[1], title='original images') 
plt.show()
