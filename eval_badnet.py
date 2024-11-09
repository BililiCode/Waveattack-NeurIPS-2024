import copy
import torch
import torchvision.transforms as transforms
import os
import random
from classifier_models import ResNet18
from config import get_arguments
from dataloader import get_dataloader
from utils import progress_bar
from image import *
from PIL import Image


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def create_bd(inputs, targets, opt):
    bd_targets = create_targets_bd(targets, opt)
    b, c, width, height = inputs.shape

    bd_inputs = copy.deepcopy(inputs)

    if opt.attack_method == 'badnets':
        bd_inputs[:,:,width-1,height-1] = 1
        bd_inputs[:,:,width-1,height-2] = 0
        bd_inputs[:,:,width-1,height-3] = 1

        bd_inputs[:,:,width-2,height-1] = 0
        bd_inputs[:,:,width-2,height-2] = 1
        bd_inputs[:,:,width-2,height-3] = 0

        bd_inputs[:,:,width-3,height-1] = 1
        bd_inputs[:,:,width-3,height-2] = 0
        bd_inputs[:,:,width-3,height-3] = 0
    elif opt.attack_method == 'blend':
        trigger_transform = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ])
        trigger = Image.open('./trigger/hello_kitty.png')
        trigger = trigger_transform(trigger).to(opt.device)
        alpha = 0.2
        bd_inputs = (1-alpha)*bd_inputs + alpha*trigger
    elif opt.attack_method == 'FTrojan':

        bd_inputs *= 255.

        bd_inputs = bd_inputs.cpu().numpy()
        bd_inputs = np.transpose(bd_inputs, (0, 2, 3, 1))
        # transfer to YUV channel
        bd_inputs = RGB2YUV(bd_inputs)

        # transfer to frequency domain
        bd_inputs = DCT(bd_inputs, opt.window_size)  # (idx, ch, w, h)

        # plug trigger frequency
        for i in range(bd_inputs.shape[0]):
            for ch in opt.channel_list:
                for w in range(0, bd_inputs.shape[2], opt.window_size):
                    for h in range(0, bd_inputs.shape[3], opt.window_size):
                        for pos in opt.pos_list:
                            bd_inputs[i][ch][w + pos[0]][h + pos[1]] += opt.magnitude

        bd_inputs = IDCT(bd_inputs, opt.window_size)  # (idx, w, h, ch)

        bd_inputs = YUV2RGB(bd_inputs)
        bd_inputs /= 255.
        bd_inputs = np.clip(bd_inputs, 0, 1)
        bd_inputs = np.transpose(bd_inputs, (0, 3, 1, 2))
        bd_inputs = torch.tensor(bd_inputs, dtype=torch.float).to(opt.device)
    elif opt.attack_method == 'none':
        pass
    else:
        raise Exception("Invalid attack method")

    return bd_inputs, bd_targets


def eval(netC, test_dl1, opt):
    print(" Eval:")

    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0

    for batch_idx, (inputs, targets) in zip(range(len(test_dl1)), test_dl1):

        inputs1, targets1 = inputs.to(opt.device), targets.to(opt.device)

        bs = inputs1.shape[0]
        total_sample += bs

        # Evaluating clean
        preds_clean = netC(inputs1)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # Evaluating backdoor
        inputs_bd, targets_bd = create_bd(inputs1, targets1, opt)

        preds_bd = netC(inputs_bd)

        correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
        total_correct_bd += correct_bd
        acc_bd = total_correct_bd * 100.0 / total_sample

        progress_bar(
            batch_idx,
            len(test_dl1),
            "Acc Clean: {:.3f} | Acc Bd: {:.3f}".format(acc_clean, acc_bd),
        )

def main():
    # Prepare arguments
    opt = get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset in ["cifar10", "cifar100"]:
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Load models and masks
    if opt.dataset in ["cifar10", 'cifar100']:
        netC = ResNet18(opt.num_classes).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    if opt.attack_method == 'FTrojan':
        opt.channel_list = [1,2]
        opt.magnitude = 30
        opt.window_size = 32
        opt.pos_list = [(31, 31), (15, 15)]

    path_model = os.path.join(
        opt.checkpoints, opt.dataset, opt.attack_mode, "{}_{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.attack_method, opt.dataset)
    )
    # path_model = os.path.join(opt.checkpoints, "ckpt_epoch_200.pth")
    state_dict = torch.load(path_model)
    print("load C")
    netC.load_state_dict(state_dict['netC'])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)

    # Prepare dataloader
    train_dl = get_dataloader(opt, train=True, preTransform=False)
    test_dl = get_dataloader(opt, train=False)

    eval(netC, test_dl, opt)

if __name__ == "__main__":
    main()
