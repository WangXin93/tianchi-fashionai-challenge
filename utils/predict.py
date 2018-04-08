# 预测结果并写入question.csv文件

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from utils.mean_std import means, stds
import time
from torch.autograd import Variable


csv_file = '/home/wangx/datasets/fashionAI/rank/Tests/question.csv'
order = ['collar_design_labels',
         'neckline_design_labels',
         'skirt_length_labels',
         'sleeve_length_labels',
         'neck_design_labels',
         'coat_length_labels',
         'lapel_design_labels',
         'pant_length_labels']


def create_question_csv(csv_file=csv_file):
    """Divide question.csv into 8 part in terms of type, and save 
    them in the same directory as csv_file.

    Args:
        csv_file (str): Path of question.csv file.

    """
    df = pd.read_csv(csv_file, names=['image', 'type', 'answer'])
    for t in df['type'].unique():
        tdf = df[df['type']==t]
        fname = csv_file.replace('question', 'question_{}'.format(t))
        tdf.to_csv(fname, index=False, header=False)


class FashionAttrsDataset(Dataset):
    """Fashion Attributes dataset for question.csv."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fashion_frame = pd.read_csv(csv_file, names=['image', 'type','answer'])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fashion_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.fashion_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {'image': image}
        return sample


def create_dataset(label_type,
                   csv_file=('/home/wangx/datasets/fashionAI/rank'
                             '/Tests/question_{}.csv'),
                   root_dir = '/home/wangx/datasets/fashionAI/rank'):
    """Create dataset, dataloader of single type for question.csv

    Args:
        label_type (str): Type of label
        csv_file: csv_file pattern for 8 attributes

    Returns:
        A dict contains image_dataset, dataloader,
        dataset_size.

    """
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means[label_type], stds[label_type])
        ])
    image_dataset = FashionAttrsDataset(csv_file.format(label_type),
                                        root_dir,
                                        data_transforms)
    dataloader = DataLoader(image_dataset,
                            batch_size=32,
                            num_workers=4)
    dataset_size = len(image_dataset)
    
    return {'image_dataset': image_dataset,
            'dataloader': dataloader,
            'dataset_size': dataset_size}


def create_datasets(label_types,
                   csv_file=('/home/wangx/datasets/fashionAI/rank'
                             '/Tests/question_{}.csv'),
                   root_dir = '/home/wangx/datasets/fashionAI/rank'):

    """Create dataloaders for 8 types
    
    Args:
        label_types (list): List of each type

    Returns:
        A dict contains image_datasets, dataloaders,
        dataset_sizes of 8 types.
    """
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for t in order:
        out = create_dataset(t, csv_file=csv_file, root_dir=root_dir)
        image_datasets[t] = out['image_dataset']
        dataloaders[t] = out['dataloader']
        dataset_sizes[t] = out['dataset_size']
    return {'image_datasets': image_datasets,
            'dataloaders': dataloaders,
            'dataset_sizes': dataset_sizes}
        

def predict_model(model, saved_model, dataloader, use_gpu):
    """Predict probabilities based on trained model

    Args:
        model: Defined model from torchvision
        saved_model (str): Path of trained model parameters
        dataloader: Dataloader of test images
        use_gpu: Where to use gpu

    Returns:
        result (list): Results of each sample

    """
    result = []

    since = time.time()

    model.train(False)  # Set model to evaluate mode
    # Load parameters
    model.load_state_dict(torch.load(saved_model))

    # Iterate over data.
    for data in dataloader:
        # get the inputs
        inputs = data['image']

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs)
        
        # Convert to cpu
        probs = probs.data.cpu().numpy()

        for p in probs:
            result.append(p)

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return result
