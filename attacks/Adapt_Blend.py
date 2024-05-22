'''
This is the implement of invisible sample-specific backdoor attack (ISSBA) [1].

Reference:
[1] Invisible Backdoor Attack with Sample-Specific Triggers. ICCV, 2021.
'''


import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
from operator import __add__
from .base import *
from collections import namedtuple
from torchvision import models as tv
import lpips
from torch.utils.data import DataLoader
import imageio
from torchvision import transforms
from torchvision.datasets import CIFAR10

from math import sqrt
from PIL import Image

class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name in ["cifar10","cifar100"] or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class GetPoisonedDataset(CIFAR10):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = torch.tensor(self.data[index])
        label = torch.tensor(self.targets[index])
        return img, label


class ProbTransform(torch.nn.Module):
    """The data augmentation transform by the probability.

    Args:
        f (nn.Module): the data augmentation transform operation.
        p (float): the probability of the data augmentation transform.
    """
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    """The data augmentation transform.

    Args:
        dataset_name (str): the name of the dataset.
    """
    def __init__(self, dataset_name):
        super(PostTensorTransform, self).__init__()
        if dataset_name == 'mnist':
            input_height, input_width = 28, 28
        elif dataset_name in ['cifar10', 'cifar100']:
            input_height, input_width = 32, 32
        elif dataset_name == 'gtsrb':
            input_height, input_width = 32, 32

        if dataset_name in ["cifar10", "cifar100"]:
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

        self.random_crop = transforms.RandomCrop(input_height, padding=4)


    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Adapt_Blend(Base):
    """Construct the backdoored model with ISSBA method.

    Args:
        dataset_name (str): the name of the dataset.
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        train_steg_set (types in support_list): Training dataset for the image steganography encoder and decoder.
        model (torch.nn.Module): Victim model.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        secret_size (int): Size of the steganography secret.
        enc_height (int): Height of the input image into the image steganography encoder.
        enc_width (int): Width of the input image into the image steganography encoder.
        enc_in_channel (int): Channel of the input image into the image steganography encoder.
        enc_total_epoch (int): Training epoch of the image steganography encoder.
        enc_secret_only_epoch (int): The final epoch to train the image steganography encoder with only secret loss function.
        enc_use_dis (bool): Whether to use discriminator during the training of the image steganography encoder. Default: False.
        encoder (torch.nn.Module): The pretrained image steganography encoder. Default: None.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """
    def __init__(self,
                 dataset_name,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 reg_rate,
                 schedule=None,
                 seed=0,
                 deterministic=False,
                 ):
        super(Adapt_Blend, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)
        self.dataset_name = dataset_name

        total_num = len(train_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        reg_num = int(total_num * reg_rate)
        assert reg_num >= 0, 'reg_num should greater than or equal to zero.'

        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
        self.poisoned_rate = poisoned_rate

        self.reg_set = frozenset(tmp_list[poisoned_num:(poisoned_num+reg_num)])
        self.reg_rate = reg_rate

        self.y_target = y_target
        self.train_poisoned_data, self.train_poisoned_label = [], []
        self.test_poisoned_data, self.test_poisoned_label = [], []

        if dataset_name in ["cifar10", "cifar100"]:
            self.normalizer = None
        elif dataset_name == "mnist":
            self.normalizer = None
        elif dataset_name == "gtsrb":
            self.normalizer = None
        else:
            self.normalizer = None

        self.pieces = 16
        self.mask_rate = 0.5
        self.masked_pieces = round(self.mask_rate * self.pieces)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        """
            Return the poisoned dataset.
        """
        if len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) == 0:
            return None, None
        elif len(self.train_poisoned_data) == 0 and len(self.test_poisoned_data) != 0:
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return None, poisoned_test_dataset
        elif len(self.train_poisoned_data) != 0 and len(self.test_poisoned_data) == 0:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            return poisoned_train_dataset, None
        else:
            poisoned_train_dataset = GetPoisonedDataset(self.train_poisoned_data, self.train_poisoned_label)
            poisoned_test_dataset = GetPoisonedDataset(self.test_poisoned_data, self.test_poisoned_label)
            return poisoned_train_dataset, poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def reset_grad(self, optimizer, d_optimizer):
        optimizer.zero_grad()
        d_optimizer.zero_grad()


    def create_targets_bd(self, targets):

        bd_targets = torch.ones_like(targets) * self.y_target

        return bd_targets.to(self.device)

    def get_trigger_mask(self, img_size, total_pieces, masked_pieces):
        div_num = sqrt(total_pieces)
        step = int(img_size // div_num)
        candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
        mask = torch.ones((img_size, img_size))
        for i in candidate_idx:
            x = int(i % div_num)  # column
            y = int(i // div_num)  # row
            mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
        return mask

    def create_bd(self, inputs, eval=False):

        b, c, width, height = inputs.shape

        bd_inputs = deepcopy(inputs)

        trigger_transform = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ])
        trigger = Image.open('../core/trigger/hellokitty_32.png')
        trigger = trigger_transform(trigger).to(self.device)

        if not eval:
            alpha = 0.15
            mask = self.get_trigger_mask(width, self.pieces, self.masked_pieces).to(self.device)
            bd_inputs = bd_inputs + alpha * mask * (trigger - bd_inputs)
        else:
            alpha = 0.2
            bd_inputs = bd_inputs + alpha * (trigger - bd_inputs)

        return bd_inputs

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        self.device = device

        self.post_transforms = None
        if self.dataset_name in ['cifar10', 'cifar100']:
            self.post_transforms = PostTensorTransform(self.dataset_name).to(self.device)

        self.work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = Log(osp.join(self.work_dir, 'log.txt'))


        self.get_img()

        trainset, testset = self.train_dataset, self.test_dataset
        train_dl = DataLoader(
            trainset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)
        test_dl = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        bd_train_save_dataset = []
        bd_train_dataset, bd_train_labset = [], []
        for idx, (img, lab) in enumerate(train_dl):
            if idx in self.poisoned_set:
                img = img.to(self.device)
                encoded_image = self.create_bd(img, eval=False)
                bd_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
                bd_train_save_dataset.append(encoded_image.cpu().detach().tolist()[0])
                bd_train_labset.append(self.y_target)
            elif idx in self.reg_set:
                img = img.to(self.device)
                encoded_image = self.create_bd(img, eval=False)
                bd_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
                bd_train_labset.append(lab.tolist()[0])
            else:
                bd_train_dataset.append(img.tolist()[0])
                bd_train_labset.append(lab.tolist()[0])

        bd_train_dl = GetPoisonedDataset(bd_train_dataset, bd_train_labset)
        torch.save(bd_train_save_dataset, osp.join(self.work_dir, "poison_data"))

        bd_test_dataset, bd_test_labset = [], []
        for idx, (img, lab) in enumerate(test_dl):
            img = img.to(self.device)
            encoded_image = self.create_bd(img, eval=True)
            bd_test_dataset.append(encoded_image.cpu().detach().tolist()[0])
            bd_test_labset.append(self.y_target)

        bd_poi_train_dataset, bd_poi_train_labset = [], []
        for idx, (img, lab) in enumerate(train_dl):
            if idx not in self.poisoned_set:
                continue
            img = img.to(self.device)
            encoded_image = self.create_bd(img, eval=False)
            bd_poi_train_dataset.append(encoded_image.cpu().detach().tolist()[0])
            bd_poi_train_labset.append(self.y_target)

        cln_test_dl = testset
        bd_test_dl = GetPoisonedDataset(bd_test_dataset, bd_test_labset)
        bd_poi_train_dl = GetPoisonedDataset(bd_poi_train_dataset, bd_poi_train_labset)

        bd_train_dl = DataLoader(
            bd_train_dl,
            batch_size=self.current_schedule['batch_size'],
            shuffle=True,
            num_workers=self.current_schedule['num_workers'],
            worker_init_fn=self._seed_worker)


        self.model = self.model.to(device)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint

        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        self.log(msg)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            loss_list = []
            for (inputs, targets) in bd_train_dl:

                if self.normalizer:
                    inputs = self.normalizer(inputs)
                if self.post_transforms:
                    inputs = self.post_transforms(inputs)

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predict_digits = self.model(inputs)
                loss = self.loss(predict_digits, targets)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + 'Train [{}] Loss: {:.4f}\n'.format(i, np.mean(loss_list))
            self.log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test result on benign test dataset
                predict_digits, labels = self._test(cln_test_dl, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
                self.log(msg)

                predict_digits, labels = self._test(bd_poi_train_dl, device, self.current_schedule['batch_size'],
                                                    self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Testing result on poisoned train dataset alpha=0.15==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct / total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct / total_num}, time: {time.time() - last_time}\n"
                self.log(msg)

                # test result on poisoned test dataset
                # if self.current_schedule['benign_training'] is False:
                predict_digits, labels = self._test(bd_test_dl, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                total_num = labels.size(0)
                prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset alpha=0.2==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                self.log(msg)

                self.model = self.model.to(device)
                self.model.train()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                self.model.eval()
                self.model = self.model.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(self.work_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model = self.model.to(device)
                self.model.train()


    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                if self.normalizer:
                    batch_img = self.normalizer(batch_img)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num} time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)

    def get_img(self, path=None):
        """Get the encoded images with the trigger pattern.

        Args:
            path (str): The path of the saved image steganography encoder.
        """

        train_dl = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            worker_init_fn=self._seed_worker)

        for _, (image_input, target) in enumerate(train_dl):
            image_input, target = image_input.cuda(), target.cuda()
            reg_image = self.create_bd(image_input, eval=False)

            poi_image = self.create_bd(image_input, eval=False)

            image_input = image_input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            reg_image = reg_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            poi_image = poi_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            # residual = residual.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            imageio.imwrite(os.path.join(self.work_dir, 'image_input.jpg'), image_input)
            imageio.imwrite(os.path.join(self.work_dir, 'reg_image.jpg'), reg_image)
            imageio.imwrite(os.path.join(self.work_dir, 'poi_image.jpg'), poi_image)
            # imageio.imwrite(os.path.join(self.work_dir, 'residual.jpg'), residual)
            break
