from utils.predict import FashionAttrsDataset, create_dataset, create_datasets, predict_model
from nose.tools import *
import torchvision
from torch import nn
import torch

csv_file = '/home/wangx/datasets/fashionAI/rank/Tests/question.csv'
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
root_dir = '/home/wangx/datasets/fashionAI/rank'


def predict_tests():
    # Test FashionAttrsDataset
    fashion_dataset = FashionAttrsDataset(csv_file, root_dir)
    assert_true(len(fashion_dataset) > 0)

    # Test create_dataset
    for t in order:
        out = create_dataset(t)
        # Test image_datset and dataset_size
        assert_true(len(out['image_dataset']) == out['dataset_size'])
        # Test dataloader
        loader = out['dataloader']
        batch = next(iter(loader))
        assert_equal(list(batch['image'].shape), [32, 3, 224, 224])

    # Test create_datasets
    out = create_datasets(order)
    assert_equal(len(out['image_datasets']), 8)
    assert_equal(len(out['dataloaders']), 8)
    assert_equal(len(out['dataset_sizes']), 8)

    # Test predict_model
    out = create_datasets(order)
    dataloaders = out['dataloaders']
    model_conv = torchvision.models.resnet34()
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, AttrKey['coat_length_labels'])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_conv = model_conv.cuda()

    saved_model = '/home/wangx/project/torchfashion/log/resnet34-transfer/coat_length_labels.pth'
    dataloader = dataloaders['coat_length_labels']
    result = predict_model(model_conv, saved_model, dataloader, use_gpu)
    assert_equal(len(result), 1453)
