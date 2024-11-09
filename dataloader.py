
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image



class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)


class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)

def get_transform(opt, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
        if opt.dataset != "mnist":
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
        if opt.dataset in ["cifar10", "cifar100"]:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

def get_dataloader(opt, train=True, preTransform=True, c=0, k=0):
    if preTransform:
        transform = get_transform(opt, train, c=c, k=k)
    else:
        transform = get_transform(opt, preTransform, c=c, k=k)
    if opt.dataset == "cifar10":
        if train:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar100":
        if train:
            dataset = torchvision.datasets.CIFAR100(opt.data_root, train, transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR100(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")

    if train:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True, drop_last=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=False, drop_last=False
        )
    return dataloader