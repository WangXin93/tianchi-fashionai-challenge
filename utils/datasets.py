import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from utils.mean_std import means, stds
# import Augmentor
import multiprocessing


class FashionAttrsDataset(Dataset):
    """Fashion Attributes dataset."""

    def __init__(self,
                 csv_file,
                 root_dir,
                 transform=None,
                 mode='index'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mode (str): Choice in '?', 'index', 'alpha'.
        """
        self.fashion_frame = pd.read_csv(csv_file, names=['image', 'type','category'])
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.fashion_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.fashion_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        category = self.fashion_frame.iloc[idx, 2]
        alpha = category
        if self.mode == '?':
            category = '?'
        elif self.mode == 'index':
            category = category.index('y')
        elif self.mode == 'alpha':
            category = category

        sample = {'image': image, 'category': category ,'alpha': alpha}
        return sample


def create_dataset(label_type, 
                   csv_file = './data/fashionAI/{}_{}.csv',
                   root_dir = '/home/wangx/datasets/fashionAI/base',
                   phase = ['train', 'test'],
                   label_mode='index',
                   shuffle=True,
                   img_size=224,
                   batch_size=32):
    """Create dataset, dataloader for train and test

    Args: label_type (str): Type of label
        csv_file (str): CSV file pattern for file indices.
        root_dir (str): Root dir based on paths in csv file.
        phase: list of str 'train' or 'test'.

    Returns:
        out (dict): A dict contains image_datasets, dataloaders,
            dataset_sizes
    """

    # Use Augmentor help produce more variation
    #p = Augmentor.Pipeline()
    #p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    #p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

    data_transforms = {
        'train': transforms.Compose([
            # p.torch_transform(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,
                                   contrast=0.2,
                                   saturation=0.2,
                                   hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(means[label_type], stds[label_type])
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(means[label_type], stds[label_type])
        ]),
    }

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for x in phase:    # ['train', 'test']
        image_datasets[x] = FashionAttrsDataset(csv_file.format(label_type, x),
                                                root_dir,
                                                data_transforms[x],
                                                mode=label_mode)

        dataloaders[x] = DataLoader(image_datasets[x],
                                    batch_size=batch_size,
                                    shuffle=shuffle and x=='train',
                                    num_workers=multiprocessing.cpu_count())

        dataset_sizes[x] = len(image_datasets[x])

    out = {'image_datasets': image_datasets,
           'dataloaders': dataloaders,
           'dataset_sizes': dataset_sizes}
    return out 
