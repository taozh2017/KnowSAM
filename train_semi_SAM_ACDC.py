import argparse
import numpy as np
import random
import torch
import os
import logging
import sys
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from dataloader.transforms import build_transforms, build_weak_strong_transforms
from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler


from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./SampleData',
                    help='Name of Experiment')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='Percentage of label quantity')
parser.add_argument('--dataset', type=str, default='/ACDC',
                    help='Name of Experiment')


parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int, default=3,
                    help='input channel of network')

parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-UNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-VNet_lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--image_size', type=int, default=256, help='image_size')
parser.add_argument('--point_nums', type=int, default=5, help='points number')
parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
parser.add_argument('-thd', type=bool, default=False, help='3d or not')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')

parser.add_argument('--mixed_iterations', type=int, default=12000,
                    help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=50000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
parser.add_argument("--sam_checkpoint", type=str, default="./sam_vit_b_01ec64.pth", help="sam checkpoint")


args = parser.parse_args()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    # model
    trainer = Trainer(args)

    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    # dataset
    data_transforms = build_weak_strong_transforms(args)
    train_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="train_acdc_list",
                                  transform=data_transforms)
    val_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split="val_acdc_list",
                                transform=data_transforms["valid_test"])

    # sampler
    total_slices = len(train_dataset)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} iterations per epoch".format(len(train_loader)))
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    # else:

    iter_num = 0
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            trainer.train(volume_batch, label_batch, iter_num)
            iter_num = iter_num + 1
            if iter_num > 0 and iter_num % 200 == 0:
                if "ACDC" not in args.dataset:
                    trainer.val(val_loader, snapshot_path, iter_num)
                else:
                    trainer.val_ACDC(val_loader, snapshot_path, iter_num)


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        torch.autograd.set_detect_anomaly(True)
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        torch.cuda.manual_seed(2024)

        snapshot_path = "./Results/results_ACDC_10/fold_" + str(fold)

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        shutil.copyfile("./train_semi_SAM_ACDC.py", snapshot_path + "/code/train_semi_SAM_ACDC.py")
        shutil.copyfile("./trainer_SGDL.py", snapshot_path + "/code/trainer.py")

        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        train(args, snapshot_path)