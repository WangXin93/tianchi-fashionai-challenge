import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from utils.mean_std import means, stds
import Augmentor


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
        category = category.index('y')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'category': category}
        return sample


def create_dataset(label_type):
    """Create dataset, dataloader for train and test

    Args:
        label_type (str): Type of label

    Returns:
        out (dict): A dict contains image_datasets, dataloaders,
            dataset_sizes
    """

    csv_file = './data/fashionAI/{}_{}.csv'
    root_dir = '/home/wangx/datasets/fashionAI/base'

#    data_transforms = {
#        'train': transforms.Compose([
#            transforms.RandomResizedCrop(224),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize(means[label_type], stds[label_type])
#        ]),
#        'test': transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),
#            transforms.ToTensor(),
#            transforms.Normalize(means[label_type], stds[label_type])
#        ]),
#    }
    # Use Augmentor help produce more variation
    p = Augmentor.Pipeline()
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=5)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    # p.skew(probability=0.5, magnitude=0.1)

    data_transforms = {
        'train': transforms.Compose([
            p.torch_transform(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(means[label_type], stds[label_type])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(means[label_type], stds[label_type])
        ]),
    }

    image_datasets = {x: FashionAttrsDataset(csv_file.format(label_type, x),
                                             root_dir,
                                             data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    out = {'image_datasets': image_datasets,
           'dataloaders': dataloaders,
           'dataset_sizes': dataset_sizes}
    return out 
