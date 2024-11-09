import os


import config
import torch
import torch.nn as nn
import torchvision
from classifier_models import ResNet18
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
    # image_HH = LL
    YH = YH[0].contiguous().view(b, c * 3, h // 2, w // 2)
    image_HH = YH[:, 6:9, :, :]

    if eval:
        residual = netG(image_HH)*100
    else:
        residual = netG(image_HH)*0.5
    # print(residual)
    encoded_image_HH = image_HH + residual
    YH[:, 6:9, :, :] = encoded_image_HH

    YH = [YH.contiguous().view(b, 3, c, h // 2, w // 2)]
    # LL = LL + residual
    bd_inputs = ifm((LL, YH))
    bd_inputs = bd_inputs.clamp(0, 1)

    return bd_inputs, bd_targets, residual



def figure_bd(inputs_cle, input_poi):

    b, c, h, w = inputs_cle.shape
    LL_cle, YH_cle = xfm(inputs_cle)
    YH_cle = YH_cle[0].contiguous().view(b, c * 3, h // 2, w // 2)

    LL_poi, YH_poi = xfm(input_poi)
    YH_poi = YH_poi[0].contiguous().view(b, c * 3, h // 2, w // 2)

    images = torch.cat((LL_cle, LL_poi), dim=2)
    file_name = "{}_LL.png".format('cifar10')
    file_path = os.path.join('temps/cifar10', file_name)
    torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    images = torch.cat((YH_cle[:,0:3,:,:], YH_poi[:,0:3,:,:]), dim=2)
    file_name = "{}_LH.png".format('cifar10')
    file_path = os.path.join('temps/cifar10', file_name)
    torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    images = torch.cat((YH_cle[:, 3:6, :, :], YH_poi[:, 3:6, :, :]), dim=2)
    file_name = "{}_HL.png".format('cifar10')
    file_path = os.path.join('temps/cifar10', file_name)
    torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    images = torch.cat((YH_cle[:, 6:9, :, :], YH_poi[:, 6:9, :, :]), dim=2)
    file_name = "{}_HH.png".format('cifar10')
    file_path = os.path.join('temps/cifar10', file_name)
    torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

# def create_cross(inputs1, inputs2, netG, opt):
#
#     b, c, h, w = inputs1.shape
#     LL2, YH2 = xfm(inputs2)
#     YH2 = YH2[0].contiguous().view(b, c * 3, h // 2, w // 2)
#     image_HH2 = YH2[:, 6:9, :, :]
#
#     residual = netG(image_HH2)
#
#     LL1, YH1 = xfm(inputs1)
#     YH1 = YH1[0].contiguous().view(b, c * 3, h // 2, w // 2)
#     image_HH1 = YH1[:, 6:9, :, :]
#     encoded_image_HH = image_HH1 + residual
#
#     YH1[:, 6:9, :, :] = encoded_image_HH
#
#     YH1 = [YH1.contiguous().view(b, 3, c, h // 2, w // 2)]
#     inputs_cross = ifm((LL1, YH1))
#     inputs_cross = inputs_cross.clamp(0, 1)
#     return inputs_cross, residual


def train_step(
    netC, netG, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, feature_r, epoch, opt
):
    netC.train()
    netG.train()
    print(" Training:")
    total = 0

    total_bd = 0
    total_clean = 0

    total_correct_clean = 0

    total_bd_correct = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs1, targets1) in zip(range(len(train_dl1)), train_dl1):
        optimizerC.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)

        bs = inputs1.shape[0]
        num_bd = int(opt.p_attack * bs)
        num_cross = int(opt.p_cross * bs)

        inputs_bd1, targets_bd, residual1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, opt)
        inputs_cross, _, residual2 = create_bd(inputs1[num_bd : num_bd+num_cross], targets1[num_bd : num_bd+num_cross], netG, opt)


        total_inputs = torch.cat((inputs_bd1, inputs_cross, inputs1[num_bd + num_cross :]), 0)

        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)
        # print(total_targets.shape)
        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)

        # Calculating diversity loss
        residual = torch.cat((residual1, residual2), dim=0)
        # loss_l2 = torch.square(residual).mean()
        loss_l2 = torch.abs(residual).mean()
        # total_loss = loss_con
        total_loss = loss_ce + 1 * loss_l2
        total_loss.backward()
        optimizerC.step()
        optimizerG.step()

        total += bs
        total_bd += num_bd
        total_clean += bs-num_bd

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd:], dim=1) == total_targets[num_bd:]
        )

        total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = total_bd_correct * 100.0 / total_bd
        # acc_cross = total_cross_correct * 100.0 / total_cross
        infor_string = "CE loss: {:.4f} - Accuracy: {:.3f} | BD Accuracy: {:.3f} | trigger size: {:.6f}".format(
            avg_loss, acc_clean, acc_bd, loss_l2.item()
        )
        progress_bar(batch_idx, len(train_dl1), infor_string)
        # if batch_idx == len(train_dl1) - 2:
        #     dir_temps = os.path.join(opt.temps, opt.dataset)
        #     if not os.path.exists(dir_temps):
        #         os.makedirs(dir_temps)
        #     b, c, h, w = inputs1.shape
        #     # HH_residual = nn.Upsample(size=(h, w), mode='nearest')(gen_res)
        #     residual = inputs_bd1 - inputs1[:num_bd]
        #     images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], inputs_bd1), dim=2))
        #     images = torch.cat((images, residual), dim=2)
        #     file_name = "{}_{}_{}_train_images.png".format(opt.dataset, opt.attack_method, opt.attack_mode)
        #     file_path = os.path.join(dir_temps, file_name)
        #     torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    schedulerC.step()
    schedulerG.step()


def finetune_step(
    netC, netG, optimizerC, schedulerC, train_dl1, feature_r, epoch, opt
):
    netC.train()
    netG.eval()
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
        num_cross = int(opt.p_cross * bs)

        inputs_bd1, targets_bd, residual1 = create_bd(inputs1[:num_bd], targets1[:num_bd], netG, opt)
        inputs_cross, _, residual2 = create_bd(inputs1[num_bd: num_bd + num_cross],
                                               targets1[num_bd: num_bd + num_cross], netG, opt)

        total_inputs = torch.cat((inputs_bd1, inputs_cross, inputs1[num_bd + num_cross:]), 0)

        total_targets = torch.cat((targets_bd, targets1[num_bd:]), 0)
        # print(total_targets.shape)
        preds = netC(total_inputs)
        loss_ce = criterion(preds, total_targets)


        # Calculating diversity loss
        residual = torch.cat((residual1, residual2), dim=0)
        loss_l2 = torch.abs(residual).mean()
        # total_loss = loss_con
        total_loss = loss_ce
        total_loss.backward()
        optimizerC.step()

        total += bs
        total_bd += num_bd
        total_clean += bs-num_bd

        total_correct_clean += torch.sum(
            torch.argmax(preds[num_bd:], dim=1) == total_targets[num_bd:]
        )

        total_bd_correct += torch.sum(torch.argmax(preds[:num_bd], dim=1) == targets_bd)
        total_loss += loss_ce.detach() * bs
        avg_loss = total_loss / total

        acc_clean = total_correct_clean * 100.0 / total_clean
        acc_bd = total_bd_correct * 100.0 / total_bd

        infor_string = "CE loss: {:.4f} - Accuracy: {:.3f} | BD Accuracy: {:.3f} | trigger size: {:.6f}".format(
            avg_loss, acc_clean, acc_bd, loss_l2.item()
        )
        progress_bar(batch_idx, len(train_dl1), infor_string)
        # if batch_idx == len(train_dl1) - 2:
        #     dir_temps = os.path.join(opt.temps, opt.dataset)
        #     if not os.path.exists(dir_temps):
        #         os.makedirs(dir_temps)
        #     b, c, h, w = inputs1.shape
        #     # HH_residual = nn.Upsample(size=(h, w), mode='nearest')(gen_res)
        #     residual = inputs_bd1 - inputs1[:num_bd]
        #     images = netG.denormalize_pattern(torch.cat((inputs1[:num_bd], inputs_bd1), dim=2))
        #     images = torch.cat((images, residual), dim=2)
        #     file_name = "{}_{}_{}_train_images.png".format(opt.dataset, opt.attack_method, opt.attack_mode)
        #     file_path = os.path.join(dir_temps, file_name)
        #     torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    schedulerC.step()

def eval(
    netC,
    netG,
    optimizerC,
    optimizerG,
    schedulerC,
    schedulerG,
    test_dl1,
    epoch,
    best_acc_clean,
    best_acc_bd,
    opt,
):
    netC.eval()
    netG.eval()
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

            inputs_bd, targets_bd, gen_res = create_bd(inputs1, targets1, netG, opt, eval=True)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total

            infor_string = "Clean Accuracy: {:.3f} | Backdoor Accuracy: {:.3f} | trigger size: {:.5f}".format(
                avg_acc_clean, avg_acc_bd, torch.abs(gen_res).mean()
            )
            progress_bar(batch_idx, len(test_dl1), infor_string)

            if batch_idx == len(test_dl1) - 2:
                dir_temps = os.path.join(opt.temps, opt.dataset)
                if not os.path.exists(dir_temps):
                    os.makedirs(dir_temps)
                b, c, h, w = inputs1.shape
                # HH_residual = nn.Upsample(size=(h, w), mode='nearest')(gen_res)
                residual = inputs_bd - inputs1
                images = netG.denormalize_pattern(torch.cat((inputs1, inputs_bd), dim=2))
                images = torch.cat((images,residual), dim=2)
                file_name = "{}_{}_{}_images.png".format(opt.dataset, opt.attack_method, opt.attack_mode)
                file_path = os.path.join(dir_temps, file_name)
                torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)
                file_name = "{}_{}_{}_res.png".format(opt.dataset, opt.attack_method, opt.attack_mode)
                file_path = os.path.join(dir_temps, file_name)
                torchvision.utils.save_image(torch.abs(gen_res), file_path, normalize=True, pad_value=1)

                figure_bd(inputs1, inputs_bd)

    print(
        " Result: Best Clean Accuracy: {:.3f} - Best Backdoor Accuracy: {:.3f} | Clean Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd, avg_acc_clean
        )
    )

    if (avg_acc_bd >= opt.asr_thread and avg_acc_clean > best_acc_clean) or \
        (avg_acc_clean > best_acc_clean - 0.1 and avg_acc_bd > best_acc_bd):
        print(" Saving!!")
        best_acc_clean = avg_acc_clean
        best_acc_bd = avg_acc_bd
        state_dict = {
            "state_dict": netC.state_dict(),
            "netG": netG.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "best_acc_clean": best_acc_clean,
            "best_acc_bd": best_acc_bd,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, "{}_{}_LH_{}_ckpt.pth.tar".format(opt.attack_mode, opt.attack_method, opt.dataset))
        torch.save(state_dict, ckpt_path)
    return best_acc_clean, best_acc_bd, epoch


def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = ResNet18(opt.num_classes).to(opt.device)
    elif opt.dataset == "cifar100":
        netC = ResNet18(opt.num_classes).to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = ResNet18(opt.num_classes).to(opt.device)
    else:
        raise Exception("Invalid dataset")

    # netG = StegaStampEncoder(opt).to(opt.device)
    netG = UNet(3, opt).to(opt.device)
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    # Prepare mask
    best_acc_clean = 0.0
    best_acc_bd = 0.0
    epoch = 1

    print("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    # train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    # test_dl2 = get_dataloader(opt, train=False)

    opt.l2 = 8

    for i in range(1, opt.n_iters+1):
        print(
            "Epoch {} - {} - {} | l2: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.l2
            )
        )
        #
        # if i % 50 == 0 and opt.l2<5:
        #     opt.l2 += 0.5

        # with torch.no_grad():
        #     feature_r = ref_f(netC, train_dl1, opt)

        if i <= 200:
            train_step(
                netC,
                netG,
                optimizerC,
                optimizerG,
                schedulerC,
                schedulerG,
                train_dl1,
                None,
                epoch,
                opt
            )
        else:
            finetune_step(
                netC,
                netG,
                optimizerC,
                schedulerC,
                train_dl1,
                None,
                epoch,
                opt
            )

        best_acc_clean, best_acc_bd, epoch = eval(
            netC,
            netG,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
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
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset in ["cifar10", 'cifar100']:
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 10
        if opt.dataset == 'cifar100':
            opt.num_classes = 100
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
        opt.num_classes = 43
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")
    train(opt)


if __name__ == "__main__":
    main()