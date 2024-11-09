
import os

import torch
from classifier_models import ResNet18
from config import get_arguments
from dataloader import get_dataloader
from networks.UNet import UNet
from utils import progress_bar


from pytorch_wavelets import DWTForward, DWTInverse


xfm = DWTForward(J=1, mode='zero', wave='haar').cuda()
ifm = DWTInverse(mode='zero', wave='haar').cuda()


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(inputs, targets, netG, opt, eval=False):
    bd_targets = create_targets_bd(targets, opt)
    b, c, h, w = inputs.shape
    LL, YH = xfm(inputs)
    YH = YH[0].contiguous().view(b, c * 3, h // 2, w // 2)
    image_HH = YH[:, 6:9, :, :]

    if eval:
        residual = netG(image_HH)*opt.lambda_2
    else:
        residual = netG(image_HH)*opt.lambda_1
    # print(residual)
    encoded_image_HH = image_HH + residual
    YH[:, 6:9, :, :] = encoded_image_HH

    YH = [YH.contiguous().view(b, 3, c, h // 2, w // 2)]
    bd_inputs = ifm((LL, YH))
    bd_inputs = bd_inputs.clamp(0, 1)

    return bd_inputs, bd_targets, residual


def eval(netC, netG, test_dl1, opt):
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
        inputs_bd, targets_bd, _ = create_bd(inputs1, targets1, netG, opt, eval=True)

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
    if opt.dataset in ["cifar10", "cifar100"]:
        netC = ResNet18(opt.num_classes).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    path_model = os.path.join(
        opt.checkpoints, opt.dataset, opt.attack_mode, "{}_{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.attack_method, opt.dataset)
    )
    state_dict = torch.load(path_model)
    print("load C")
    netC.load_state_dict(state_dict["state_dict"])
    netC.to(opt.device)
    netC.eval()
    netC.requires_grad_(False)
    print("load G")
    netG = UNet(3, opt).to(opt.device)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)

    # Prepare dataloader
    train_dl = get_dataloader(opt, train=True, preTransform=False)
    test_dl = get_dataloader(opt, train=False)
    eval(netC, netG, test_dl, opt)



if __name__ == "__main__":
    main()
