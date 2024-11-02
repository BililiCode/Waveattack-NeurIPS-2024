'''
This is the implement of BadNets [1].

Reference:
[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
'''

import copy
import random
import cv2
import numpy as np
import PIL
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose

from .base import *


def RGB2YUV(x_rgb):
    img = cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    return img

def YUV2RGB(x_yuv):
    img = cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return img


def DCT(x_train, window_size):
    # x_train: (w, h, ch)
    x_dct = np.zeros((x_train.shape[2], x_train.shape[0], x_train.shape[1]), dtype=np.float)
    x_train = np.transpose(x_train, (2, 0, 1))

    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float))
                x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct            # x_dct: (idx, ch, w, h)


def IDCT(x_train, window_size):
    # x_train: (ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=np.float)

    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float))
                x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct

class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, bd_inputs):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """

        bd_inputs = bd_inputs.cpu().numpy()
        bd_inputs = np.transpose(bd_inputs, (1, 2, 0))      # C,H,W->H.W.C
        # transfer to YUV channel
        bd_inputs = RGB2YUV(bd_inputs)

        # transfer to frequency domain
        bd_inputs = DCT(bd_inputs, self.window_size)  # (idx, ch, w, h)

        # plug trigger frequency
        for ch in self.channel_list:
            for w in range(0, bd_inputs.shape[1], self.window_size):
                for h in range(0, bd_inputs.shape[2], self.window_size):
                    for pos in self.pos_list:
                        bd_inputs[ch][w + pos[0]][h + pos[1]] += self.magnitude

        bd_inputs = IDCT(bd_inputs, self.window_size)  # (idx, w, h, ch)

        bd_inputs = YUV2RGB(bd_inputs)

        return bd_inputs


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, weight):
        super(AddDatasetFolderTrigger, self).__init__()

        self.channel_list = [1, 2]
        self.magnitude = 30
        self.window_size = 32
        self.pos_list = [(31, 31), (15, 15)]

    def __call__(self, img):

        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img)
        return img


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self):
        super(AddCIFAR10Trigger, self).__init__()

        self.channel_list = [1, 2]
        self.magnitude = 20
        self.window_size = 32
        self.pos_list = [(31, 31), (15, 15)]

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img)
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger())

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class PoisonedCIFAR100(CIFAR100):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR100, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger())

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == MNIST:
        return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
    elif class_name == CIFAR100:
        return PoisonedCIFAR100(benign_dataset, y_target, poisoned_rate, pattern, weight, poisoned_transform_index,
                               poisoned_target_transform_index)
    else:
        raise NotImplementedError


class FTrojan(Base):
    """Construct poisoned datasets with BadNets method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W).
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 pattern=None,
                 weight=None,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        assert pattern is None or (isinstance(pattern, torch.Tensor) and ((0 < pattern) & (pattern < 1)).sum() == 0), 'pattern should be None or 0-1 torch.Tensor.'

        super(FTrojan, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            y_target,
            poisoned_rate,
            pattern,
            weight,
            poisoned_transform_train_index,
            poisoned_target_transform_index)

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            pattern,
            weight,
            poisoned_transform_test_index,
            poisoned_target_transform_index)

        self.channel_list = [1, 2]
        self.magnitude = 30
        self.window_size = 32
        self.pos_list = [(31, 31), (15, 15)]

    def create_bd(self, inputs):

        b, c, width, height = inputs.shape

        bd_inputs = np.zeros(inputs.shape, dtype=np.float)
        for i in range(inputs.shape[0]):

            bd_input = copy.deepcopy(inputs[i])

            bd_input *= 255.

            bd_input = bd_input.cpu().numpy()
            bd_input = np.transpose(bd_input, (1, 2, 0))  # C,H,W->H.W.C
            # transfer to YUV channel
            bd_input = RGB2YUV(bd_input)

            # transfer to frequency domain
            bd_input = DCT(bd_input, self.window_size)  # (idx, ch, w, h)

            # plug trigger frequency
            for ch in self.channel_list:
                for w in range(0, bd_input.shape[1], self.window_size):
                    for h in range(0, bd_input.shape[2], self.window_size):
                        for pos in self.pos_list:
                            bd_input[ch][w + pos[0]][h + pos[1]] += self.magnitude

            bd_input = IDCT(bd_input, self.window_size)  # (idx, w, h, ch)

            bd_input = YUV2RGB(bd_input)
            bd_input = bd_input.astype(np.float32)
            bd_input /= 255.
            bd_input = np.clip(bd_input, 0, 1)
            bd_input = np.transpose(bd_input, (2, 0, 1))
            bd_inputs[i] = bd_input

        bd_inputs = torch.tensor(bd_inputs, dtype=torch.float)
        bd_inputs = bd_inputs.to(inputs.device)
        return bd_inputs

    def eval_trigger(self):
        print(" Eval Trigger:")

        total_iss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_l2 = 0.0
        total_cnt = 0

        model = inception_v3(pretrained=True, transform_input=False).cuda()
        model.eval()

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=6,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        device = torch.device("cuda:0")

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
                inputs, targets1 = inputs.to(device), targets.to(device)

                bs = inputs.shape[0]
                # Evaluating backdoor
                inputs_bd = self.create_bd(inputs)

                total_iss += IS_score(inputs, inputs_bd, model)

                img = torch.clamp(inputs * 255, min=0, max=255).byte().permute(0, 2, 3, 1).cpu().numpy()
                img_poison = torch.clamp(inputs_bd * 255, min=0, max=255).byte().permute(0, 2, 3, 1).cpu().numpy()

                psnr, ssim, iss, l2 = get_visual_values(img, img_poison)

                total_iss += iss
                total_psnr += psnr
                total_ssim += ssim
                total_l2 += l2
                total_cnt += inputs.shape[0]

        print(
            "ISS: {:.4f}  | PSNR Acc: {:.4f}  | SSIM: {:.4f} | L2: {:.4f}".format(
                total_iss / total_cnt, total_psnr / total_cnt, total_ssim / total_cnt, total_l2 / total_cnt
            )
        )

    def get_img(self, path=None):
        """Get the encoded images with the trigger pattern.

        Args:
            path (str): The path of the saved image steganography encoder.
        """

        train_dl = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        for _, (image_input, target) in enumerate(train_dl):
            image_input, target = image_input.cuda(), target.cuda()

            poi_image = self.create_bd(image_input)

            image_input = image_input.detach().cpu()
            poi_image = poi_image.detach().cpu()

            images = torch.cat((image_input, poi_image), dim=2)

            torchvision.utils.save_image(images, '../FTrojan.png', normalize=True,
                                         pad_value=1)
            # imageio.imwrite(os.path.join(self.work_dir, 'residual.jpg'), residual)
            break