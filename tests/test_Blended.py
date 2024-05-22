'''
This is the test code of benign training and poisoned training under Blended Attack.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, RandomHorizontalFlip, RandomCrop
import core


global_seed = 666
deterministic = True
torch.manual_seed(global_seed)

# Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR100
#
# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip(p=0.5),
#     RandomCrop(32, padding=4)
# ])
# trainset = dataset('../datasets', train=True, transform=transform_train, download=True)
#
# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('../datasets', train=False, transform=transform_test, download=True)
#
#
# # Show an Example of Benign Training Samples
#
# # Settings of Pattern and Weight
# '''
# pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
# pattern[0, -3:, -3:] = 255
# weight = torch.zeros((1, 32, 32), dtype=torch.float32)
# weight[0, -3:, -3:] = 0.2
# '''
# pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
# pattern[0, -3:, -3:] = 255
# weight = torch.zeros((1, 28, 28), dtype=torch.float32)
# weight[0, -3:, -3:] = 0.2
#
#
# blended = core.Blended(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.resnet20(100),
#     # model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     poisoned_transform_train_index=1,
#     poisoned_transform_test_index=1,
#     pattern=pattern,
#     weight=weight,
#     y_target=5,
#     poisoned_rate=0.05,
#     seed=global_seed,
#     deterministic=deterministic
# )
#
#
# # Train Benign Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '0',
#     'GPU_num': 1,
#
#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 4,
#
#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [100, 150],
#
#     'epochs': 200,
#
#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 100,
#
#     'save_dir': 'experiments',
#     'experiment_name': 'resnet20_CIFAR100_Blended'
# }
#
# blended.train(schedule)
# benign_model = blended.get_model()
#
#
# # Test Benign Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '1',
#     'GPU_num': 1,
#
#     'batch_size': 128,
#     'num_workers': 4,
#
#     'save_dir': 'experiments',
#     # 'experiment_name': 'test_benign_CIFAR10_Blended'
#     'experiment_name': 'test_benign_MNIST_Blended'
# }
# blended.test(test_schedule)

transform_train = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])

transform_test = Compose([
    ToPILImage(),
    Resize((32, 32)),
    ToTensor()
])

trainset = DatasetFolder(
    root='~/YZH/NAD/data/GTSRB/train', # please replace this with path to your training set
    loader=cv2.imread,
    extensions=('ppm',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None)

testset = DatasetFolder(
    root='~/YZH/NAD/data/GTSRB/test', # please replace this with path to your test set
    loader=cv2.imread,
    extensions=('ppm',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None)


# Show an Example of Benign Training Samples

# Settings of Pattern and Weight
'''
pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2
'''
pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 28, 28), dtype=torch.float32)
weight[0, -3:, -3:] = 0.2


blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.resnet20(43),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    poisoned_transform_train_index=3,
    poisoned_transform_test_index=3,
    pattern=pattern,
    weight=weight,
    y_target=5,
    poisoned_rate=0.05,
    seed=global_seed,
    deterministic=deterministic
)


# Train Benign Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gamma': 0.1,
    'schedule': [40, 80],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 100,

    'save_dir': 'experiments',
    'experiment_name': 'resnet20_GTSRB_Blended'
}

blended.train(schedule)