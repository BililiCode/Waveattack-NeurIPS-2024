import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=1e-3)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerG_milestones", type=list, default=[100, 150, 400, 500])
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 150, 300, 400])
    parser.add_argument("--schedulerG_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=100)
    parser.add_argument("--num_workers", type=float, default=1)

    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_method", type=str, default="waveAttack", help="all2one or all2all")
    parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
    parser.add_argument("--p_attack", type=float, default=0.05)
    parser.add_argument("--p_cross", type=float, default=0.05)
    parser.add_argument("--asr_thread", type=float, default=99.8)

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    return parser
