
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize, RandomCrop

import core

if __name__ == '__main__':
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

    attack = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.resnet20(10),
        loss=nn.CrossEntropyLoss(),
        y_target=5,
        poisoned_rate=0.00,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic
    )

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
        'experiment_name': 'resnet20_CIFAR10_None'
    }

    attack.eval_trigger()