
from medpy import metric
from scipy.ndimage import zoom


def getLargestCC(segmentation):
    from skimage.measure import label
    labels = label(segmentation)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    else:
        largestCC = segmentation
    return largestCC

def calculate_metric_percase(sam_pred, SGDL_pred, gt):
    sam_pred[sam_pred > 0] = 1
    SGDL_pred[SGDL_pred > 0] = 1
    gt[gt > 0] = 1
    dice_res = []
    if sam_pred.sum() > 0:
        dice_res.append(metric.binary.dc(sam_pred, gt))
    else:
        dice_res.append(0)

    if SGDL_pred.sum() > 0:
        dice_res.append(metric.binary.dc(SGDL_pred, gt))
    else:
        dice_res.append(0)

    return dice_res


def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


def test_single_volume(args, image, label, sam_model, SGDL):
    classes = args.num_classes
    patch_size = [args.image_size, args.image_size]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    sam_prediction = np.zeros_like(label)
    SGDL_prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input = input.repeat(1,3,1,1)
        with torch.no_grad():
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL(input)
            image_embeddings = sam_model.image_encoder(input)
            points_embedding, boxes_embedding, mask_embedding = sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(args.image_size / 4), int(args.image_size / 4)),
                device=args.device)
            with torch.no_grad():
                for i in range(args.num_classes):
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        # points=points_embedding[i].unsqueeze(0),
                        points=None,
                        # boxes=None,
                        boxes=boxes_embedding[i],
                        # masks=mask_embedding[i],
                        masks=F.interpolate(fusion_map[:, i, ...].unsqueeze(1).clone().detach(), size=(64, 64),
                                            mode='bilinear'),
                        # masks=None,
                    )
                    low_res_masks, iou_predictions = sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

            pred_sam = F.interpolate(low_res_masks_all, size=(args.image_size, args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)

            out_SGDL = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().detach().numpy()
            out_sam = torch.argmax(pred_sam_soft, dim=1).squeeze(0).cpu().detach().numpy()

            pred_SGDL = zoom(out_SGDL, (x / patch_size[0], y / patch_size[1]), order=0)
            pred_sam = zoom(out_sam, (x / patch_size[0], y / patch_size[1]), order=0)

            SGDL_prediction[ind] = pred_SGDL
            sam_prediction[ind] = pred_sam

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(sam_prediction == i, SGDL_prediction == i, label == i))
    return metric_list


if __name__ == '__main__':
    import cv2
    import torch
    import argparse

    import torch.nn.functional as F
    from dataloader.dataset import build_Dataset
    from dataloader.transforms import build_transforms
    from torch.utils.data import DataLoader
    import numpy as np

    from Model.model import KnowSAM

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./SampleData',
                        help='Name of Experiment')
    parser.add_argument('--dataset', type=str, default='/ACDC',
                        help='Name of Experiment')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=list, default=256,
                        help='patch size of network input')
    parser.add_argument('--point_nums', type=int, default=5, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('--thd', type=bool, default=False, help='3d or not')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")

    parser.add_argument('--sam_model_path', type=str,
                        default="./sam_best_model.pth",
                        help='model weight path')
    parser.add_argument('--SGDL_model_path', type=str,
                        default="./SGDL_iter_16400.pth",
                        help='model weight path')

    args = parser.parse_args()
    data_transforms = build_transforms(args)

    test_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="test_list",
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = "SGDL"
    if model == "SGDL":
        SGDL_model = KnowSAM(args, bilinear=True).to(args.device).train()
        SGDL_checkpoint = torch.load(args.SGDL_model_path)
        SGDL_model.load_state_dict(SGDL_checkpoint)
        SGDL_model.eval()

        avg_dice_list = 0.0
        avg_iou_list = 0.0
        avg_hd95_list = 0.0
        avg_asd_list = 0.0
        classes = args.num_classes
        patch_size = [args.image_size, args.image_size]
        final_res = [0, 0, 0, 0, 0]

        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image, label = test_image.squeeze(0).cpu().detach().numpy(), test_label.squeeze(0).cpu().detach().numpy()
            SGDL_prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                input = input.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL_model(input)
                    fusion_map_soft = torch.softmax(fusion_map, dim=1)
                    out_SGDL = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().detach().numpy()
                    pred_SGDL = zoom(out_SGDL, (x / patch_size[0], y / patch_size[1]), order=0)
                    SGDL_prediction[ind] = pred_SGDL

            metric_list = []
            for i in range(1, classes):
                disc_pred = SGDL_prediction == i
                gt = label == i
                disc_pred[disc_pred > 0] = 1
                if 1:
                    disc_pred = getLargestCC(disc_pred)
                gt[gt > 0] = 1
                single_class_res = []
                if disc_pred.sum() > 0:
                    single_class_res.append(metric.binary.dc(disc_pred, gt))
                    single_class_res.append(metric.binary.jc(disc_pred, gt))
                    single_class_res.append(metric.binary.asd(disc_pred, gt))
                    single_class_res.append(metric.binary.hd95(disc_pred, gt))
                else:
                    single_class_res = [0, 0, 0, 0, 0]
                metric_list.append(single_class_res)

            metric_list = np.array(metric_list).astype("float32")
            metric_list = np.mean(metric_list, axis=0)

            print(metric_list)
            final_res += metric_list
        final_res = [x / len(test_loader) for x in final_res]
        print("avg_dice: ", final_res[0])
        print("avg_iou: ", final_res[1])
        print("avg_asd: ", final_res[2])
        print("avg_hd95: ", final_res[3])




