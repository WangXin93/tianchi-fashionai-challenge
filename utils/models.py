# 创建待训练的模型
import torchvision
from torch import nn

def create_model(model_key,
                 pretrained,
                 num_of_classes,
                 use_gpu):
    """Create CNN model

    Args:
        model_key (str): Name of the model to be created.
        pretrained: If True, only train the laster layer.
        num_of_classes (int): Number of categories of outputs.

    Returns:
        model_conv: A defined model to be train

    Raises:
        ValueError: Error while asking an unrecognized model.
    """
    if model_key == 'resnet18':
        model_conv = torchvision.models.resnet18(pretrained=True)
    elif model_key == 'resnet34':
        model_conv = torchvision.models.resnet34(pretrained=True)
    elif model_key == 'resnet50':
        model_conv = torchvision.models.resnet50(pretrained=True)
    elif model_key == 'resnet101':
        model_conv = torchvision.models.resnet101(pretrained=True)
    elif model_key == 'inception_v3':
        model_conv = torchvision.models.inception_v3(pretrained=True)
    else:
        raise ValueError("Unrecognized name of model {}".format(model_key))

    if pretrained:
        # Lock parameters for transfer learning
        for param in model_conv.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_of_classes)

    # Initialize newly added module parameters
    nn.init.xavier_uniform(model_conv.fc.weight)
    nn.init.constant(model_conv.fc.bias, 0)

    if use_gpu:
        model_conv = model_conv.cuda()

    return model_conv
