import copy
import os


import config
import torch
import torchvision.transforms as transforms
from classifier_models import ResNet18
from dataloader import get_dataloader
from utils import progress_bar
from PIL import Image
from image import *
from torch import nn

def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all_mask":
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
    else:
        raise Exception("Invalid attack method")

    return bd_inputs, bd_targets

def train_step(
    netC, optimizerC, schedulerC, train_dl1, epoch, opt
):
    netC.train()
    print(" Training:")
    total = 0

    total_bd = 0
    total_clean = 0

    total_correct_clean = 0
    total_bd_correct = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs1, targets1) in zip(range(len(train_dl1)), train_dl1):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(opt.p_attack * bs)

        inputs_bd, targets_bd = create_bd(inputs1[:num_bd], targets1[:num_bd], opt)

        total_inputs = torch.cat((inputs_bd, inputs1[num_bd:]), 0)
        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)

        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        total_loss = loss_ce
        total_loss.backward()
        optimizerC.step()

        total += bs
        total_bd += num_bd
        total_clean += bs - num_bd

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd:], dim=1) == total_targets[num_bd:]
        )
        total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = total_bd_correct * 100.0 / total_bd

        infor_string = "CE loss: {:.4f} - Accuracy: {:.3f} | BD Accuracy: {:.3f}".format(
            avg_loss, acc_clean, acc_bd
        )
        progress_bar(batch_idx, len(train_dl1), infor_string)

    schedulerC.step()

def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl1,
    epoch,
    best_acc_clean,
    best_acc_bd,
    opt,
):
    netC.eval()
    print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_bd = 0.0

    for batch_idx, (inputs1, targets1) in zip(range(len(test_dl1)), test_dl1):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            bs = inputs1.shape[0]

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
            total_correct_clean += correct_clean

            inputs_bd, targets_bd = create_bd(inputs1, targets1, opt)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total

            infor_string = "Clean Accuracy: {:.3f} | Backdoor Accuracy: {:.3f}".format(
                avg_acc_clean, avg_acc_bd
            )
            progress_bar(batch_idx, len(test_dl1), infor_string)

    print(
        " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f} | Clean Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd, avg_acc_clean
        )
    )
    if avg_acc_clean > best_acc_clean or (avg_acc_clean > best_acc_clean - 0.1 and avg_acc_bd > best_acc_bd):
        print(" Saving!!")
        best_acc_clean = avg_acc_clean
        best_acc_bd = avg_acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "best_acc_clean": best_acc_clean,
            "best_acc_bd": best_acc_bd,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.attack_method, opt.dataset))
        torch.save(state_dict, ckpt_path)
    return best_acc_clean, best_acc_bd, epoch

def train(opt):
    # Prepare model related things
    if opt.dataset in ["cifar10", 'cifar100']:
        netC = ResNet18(opt.num_classes).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    # Continue training ?
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.attack_method, opt.dataset))
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        epoch = state_dict["epoch"] + 1
        optimizerC.load_state_dict(state_dict["optimizerC"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        best_acc_clean = state_dict["best_acc_clean"]
        best_acc_bd = state_dict["best_acc_bd"]
        best_acc_cross = state_dict["best_acc_cross"]
        opt = state_dict["opt"]
        print("Continue training")
    else:
        # Prepare mask
        best_acc_clean = 0.0
        best_acc_bd = 0.0

        epoch = 1

        # Reset tensorboard
        print("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)

    for i in range(opt.n_iters):
        print(
            "Epoch {} - {} - {}:".format(
                epoch, opt.dataset, opt.attack_mode
            )
        )
        train_step(
            netC,
            optimizerC,
            schedulerC,
            train_dl1,
            epoch,
            opt,
        )

        best_acc_clean, best_acc_bd, epoch = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl1,
            epoch,
            best_acc_clean,
            best_acc_bd,
            opt,
        )
        epoch += 1
        if epoch > opt.n_iters:
            break


def main():
    opt = config.get_arguments().parse_args()
    if opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == 'cifar100':
        opt.num_classes = 100
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset in ["cifar10", 'cifar100']:
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    if opt.attack_method == 'FTrojan':
        opt.channel_list = [1,2]
        opt.magnitude = 20
        opt.window_size = 32
        opt.pos_list = [(31, 31), (15, 15)]

    train(opt)


if __name__ == "__main__":
    main()
