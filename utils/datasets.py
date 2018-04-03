import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image


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


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


csv_file = '/home/wangx/datasets/fashionAI/web/Annotations/skirt_length_labels_{}.csv'
root_dir = '/home/wangx/datasets/fashionAI/web/'


image_datasets = {x: FashionAttrsDataset(csv_file.format(x), root_dir, data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
