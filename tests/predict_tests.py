from utils.predict import predict_model
from utils.datasets import FashionAttrsDataset, create_dataset
from nose.tools import *
import torchvision
from torch import nn
import torch

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


def predict_tests():
    # Test FashionAttrsDataset
    csv_file = '/home/wangx/datasets/fashionAI/rank/Tests/question.csv'
    root_dir = '/home/wangx/datasets/fashionAI/rank'
    fashion_dataset = FashionAttrsDataset(csv_file,
                                          root_dir,
                                          mode='?')
    assert_true(len(fashion_dataset) > 0)

    # Test create_dataset
    csv_file='/home/wangx/project/torchfashion/questions/{}_{}.csv'
    root_dir = '/home/wangx/datasets/fashionAI/rank'
    for t in order:
        out = create_dataset(label_type=t,
                             csv_file=csv_file,
                             root_dir=root_dir,
                             phase=['test'],
                             label_mode='?')
        # Test image_datset and dataset_size
        assert_true(len(out['image_datasets']['test']) == out['dataset_sizes']['test'])
        # Test dataloader
        loader = out['dataloaders']['test']
        batch = next(iter(loader))
        assert_equal(list(batch['image'].shape), [32, 3, 224, 224])

    # Test predict_model
    out = create_dataset('coat_length_labels',
                          csv_file=csv_file,
                          root_dir=root_dir,
                          phase=['test'],
                          label_mode='?')
    dataloader = out['dataloaders']['test']
    model_conv = torchvision.models.resnet34()
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, AttrKey['coat_length_labels'])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_conv = model_conv.cuda()

    saved_model = './log/resnet34-transfer/coat_length_labels.pth'
    result = predict_model(model_conv, saved_model, dataloader, use_gpu)
    assert_equal(len(result), 1453)
