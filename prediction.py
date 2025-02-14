import cv2
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import eval
from Model.model import KnowSAM


def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


from skimage.measure import label
def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 2):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./SampleData',
                        help='Name of Experiment')

    parser.add_argument('--dataset', type=str, default='/tumor_1',
                        help='Name of Experiment')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=list, default=256,
                        help='patch size of network input')
    parser.add_argument('--point_nums', type=int, default=10, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('--thd', type=bool, default=False, help='3d or not')

    parser.add_argument('--sam_model_path', type=str,
                        default="./Results/Result_tumor_10/fold_0/sam_best_model.pth",
                        help='model weight path')

    parser.add_argument('--SGDL_model_path', type=str,
                        default="./Results/Result_tumor_10/fold_0/SGDL_best_model.pth",
                        help='model weight path')

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    bilinear = True
    Largest = False
    data_transforms = build_transforms(args)

    test_dataset_list = ["test_CVC-300", "test_CVC-ClinicDB",]

    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(args, data_dir=args.data_path + args.dataset, split=test_dataset_name,
                                     transform=data_transforms["valid_test"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        # print(args.SGDL_model_path)
        SGDL_model = KnowSAM(args, bilinear=bilinear).to(args.device).train()
        SGDL_checkpoint = torch.load(args.SGDL_model_path)
        SGDL_model.load_state_dict(SGDL_checkpoint)
        SGDL_model.eval()

        avg_dice_list = []
        avg_hd95_list = []
        avg_iou_list = []
        avg_sp_list = []
        avg_se_list = []
        avg_prec_list = []
        avg_recall_list = []
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label, ori_image = sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), sampled_batch["ori_image"].cuda()
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL_model(test_image)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)

            if Largest:
                pseudo_label = torch.argmax(fusion_map_soft, dim=1)
                fusion_map_soft = get_ACDC_2DLargestCC(pseudo_label).unsqueeze(0)

            eval_list = eval(test_label, fusion_map_soft, thr=0.5)

            avg_dice_list.append(eval_list[0])
            avg_iou_list.append(eval_list[1])
            avg_hd95_list.append(eval_list[2])

        avg_dice = np.mean(avg_dice_list)
        avg_hd95 = np.mean(avg_hd95_list)
        avg_iou = np.mean(avg_iou_list)

        print(test_dataset_name, " :")
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)
        print("avg_hd95: ", avg_hd95)
