'''
This is the test code of poisoned training on GTSRB, CIFAR10, MNIST, using dataset class of torchvision.datasets.DatasetFolder torchvision.datasets.CIFAR10 torchvision.datasets.MNIST.
The attack method is Adapt-Blend.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, CIFAR100, MNIST
import core


global_seed = 666
deterministic = False
torch.manual_seed(global_seed)


def read_image(img_path, type=None):
    img = cv2.imread(img_path)
    if type is None:        
        return img
    elif isinstance(type,str) and type.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(type,str) and type.upper() == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError


class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label


# ===== Train backdoored model on GTSRB using with GTSRB ===== 

# transform_train = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor()
# ])
#
# transform_test = Compose([
#     ToPILImage(),
#     Resize((32, 32)),
#     ToTensor()
# ])
#
# trainset = DatasetFolder(
#     root='~/YZH/NAD/data/GTSRB/train', # please replace this with path to your training set
#     loader=cv2.imread,
#     extensions=('ppm',),
#     transform=transform_train,
#     target_transform=None,
#     is_valid_file=None)
#
# testset = DatasetFolder(
#     root='~/YZH/NAD/data/GTSRB/test', # please replace this with path to your test set
#     loader=cv2.imread,
#     extensions=('ppm',),
#     transform=transform_test,
#     target_transform=None,
#     is_valid_file=None)
#
#
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,
#
#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 8,
#
#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 1e-4,
#     'gamma': 0.1,
#     'schedule': [40, 80],
#
#     'epochs': 100,
#
#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 100,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'resnet20_GTSRB_Adapt_Blend'
# }
#
#
# # Configure the attack scheme
# Adapt_Blend = core.Adapt_Blend(
#     dataset_name="gtsrb",
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.resnet20(43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=5,
#     poisoned_rate=0.05,      # follow the default configure in the original paper
#     reg_rate = 0.05,
#     schedule=schedule,
#     seed=global_seed,
#     deterministic=deterministic
# )
#
# Adapt_Blend.train(schedule=schedule)

# ===== Train backdoored model on GTSRB using with GTSRB (done) ===== 


# ===== Train backdoored model on CIFAR10 using with CIFAR10 ===== 

# Prepare datasets and follow the default data augmentation in the original paper
transform_train = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
])
transform_test = Compose([
    transforms.Resize((32, 32)),
    ToTensor(),
])

trainset = CIFAR10(
    root='../datasets/', # please replace this with path to your dataset
    transform=transform_train,
    target_transform=None,
    train=True,
    download=True)
testset = CIFAR10(
    root='../datasets', # please replace this with path to your dataset
    transform=transform_test,
    target_transform=None,
    train=False,
    download=True)


schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 150],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'resnet20_CIFAR10_Adapt_Blend'
}


# Configure the attack scheme
Adapt_Blend = core.Adapt_Blend(
    dataset_name="cifar10",
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.resnet20(10),
    loss=nn.CrossEntropyLoss(),
    y_target=5,
    poisoned_rate=0.05,      # follow the default configure in the original paper
    reg_rate = 0.05,
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)

Adapt_Blend.train(schedule=schedule)
