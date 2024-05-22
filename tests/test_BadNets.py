'''
This is the test code of poisoned training under BadNets.
'''


import os.path as osp

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop

import core


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '0'
datasets_root_dir = '../datasets'


# ========== ResNet-18_CIFAR-10_BadNets ==========
dataset = torchvision.datasets.CIFAR10

transform_train = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(32, padding=4),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

pattern = torch.zeros((32, 32), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((32, 32), dtype=torch.float32)
weight[-3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.VGG(vgg_name="VGG16", num_classes=10),
    loss=nn.CrossEntropyLoss(),
    y_target=5,
    poisoned_rate=0.00,
    pattern=pattern,
    weight=weight,
    seed=global_seed,
    deterministic=deterministic
)

# Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/CIFAR/train_watermarked.py)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': True,
    'batch_size': 128,
    'num_workers': 2,

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
    'experiment_name': 'VGG16_CIFAR10_None'
}
badnets.train(schedule)


# ========== ResNet-18_GTSRB_BadNets ==========
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
# pattern = torch.zeros((32, 32), dtype=torch.uint8)
# pattern[-3:, -3:] = 255
# weight = torch.zeros((32, 32), dtype=torch.float32)
# weight[-3:, -3:] = 1.0
#
# badnets = core.BadNets(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.resnet20(43),
#     loss=nn.CrossEntropyLoss(),
#     y_target=5,
#     poisoned_rate=0.00,
#     pattern=pattern,
#     weight=weight,
#     poisoned_transform_train_index=2,
#     poisoned_transform_test_index=2,
#     seed=global_seed,
#     deterministic=deterministic
# )
#
# # Train Attacked Model (schedule is the same as https://github.com/THUYimingLi/Open-sourced_Dataset_Protection/blob/main/GTSRB/train_watermarked.py)
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
#     'GPU_num': 1,
#
#     'benign_training': True,
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
#     'save_epoch_interval': 10,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'resnet20_GTSRB_None'
# }
# badnets.train(schedule)
