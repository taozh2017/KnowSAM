import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.sam.build_sam import sam_model_registry
import torch.optim as optim
from utils.losses import dice_loss, loss_diff1, loss_diff2, KDLoss, DiceLoss
import logging
from utils.utils import dice_coef

import numpy as np

from Model.model import KnowSAM
from prediction_ACDC import test_single_volume

ce_loss = torch.nn.CrossEntropyLoss()

GPUdevice = torch.device('cuda', 0)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.criterion_mse = nn.MSELoss()
        self.KDLoss = KDLoss(T=10)
        self.dice_loss = DiceLoss(args.num_classes)

        self.sam_model = sam_model_registry[args.model_type](args).to(args.device).train()
        self.SGDL = KnowSAM(args).cuda().train()

        self.optimizer_sam = optim.Adam(self.sam_model.parameters(), lr=args.lr)
        self.optimizer_SGDL = torch.optim.SGD(self.SGDL.parameters(), lr=args.UNet_lr, momentum=0.9,
                                              weight_decay=0.0001)

        self.best_performance_sam = 0.0
        self.best_performance_SGDL = 0.0

        for n, value in self.sam_model.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            elif "super_prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def entropy_loss(self, p, C=2):
        # p N*C*W*H*D
        y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
             torch.tensor(np.log(C)).cuda()
        ent = torch.mean(y1)
        return ent

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)

    def mix_up(self, fusion_map_soft, volume_batch, pseudo_label, labeled_label, consistency_weight, patch_size=4,
               top_k=5):
        unlabel_pseudo_label = torch.argmax(pseudo_label.clone(), dim=1)
        entropy_unlab = self.get_entropy_map(fusion_map_soft[self.args.labeled_bs:])
        entropy_lab = self.get_entropy_map(fusion_map_soft[:self.args.labeled_bs])
        pooling = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        entropy_unlab = pooling(entropy_unlab).view(self.args.labeled_bs, -1)
        entropy_lab = pooling(entropy_lab).view(self.args.labeled_bs, -1)

        _, min_indices_flat = torch.topk(entropy_unlab, top_k, largest=False)
        min_indices_2d = torch.stack([min_indices_flat // patch_size, min_indices_flat % patch_size], dim=-1)
        _, min_indices_flat_lab = torch.topk(entropy_lab, top_k, largest=False)
        min_indices_2d_lab = torch.stack([min_indices_flat_lab // patch_size, min_indices_flat_lab % patch_size],
                                         dim=-1)

        labeled_volume_batch = volume_batch[:self.args.labeled_bs]
        unlabeled_volume_batch = volume_batch[self.args.labeled_bs:]

        unlabeled_volume_batch_mix = torch.zeros_like(unlabeled_volume_batch).cuda()
        unlabel_pseudo_label_mix = torch.zeros_like(unlabel_pseudo_label).cuda()
        labeled_volume_batch_mix = torch.zeros_like(labeled_volume_batch).cuda()
        labeled_pseudo_label_mix = torch.zeros_like(labeled_label).cuda()

        patch_h = int(self.args.image_size / patch_size)
        for b in range(self.args.labeled_bs):
            index = min_indices_2d[b]
            img_mask = torch.zeros((self.args.image_size, self.args.image_size)).cuda()
            index_lab = min_indices_2d_lab[b]
            img_mask_lab = torch.zeros((self.args.image_size, self.args.image_size)).cuda()
            for n in index:
                img_mask[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1
            for n in index_lab:
                img_mask_lab[n[0] * patch_h: (n[0] + 1) * patch_h, n[1] * patch_h: (n[1] + 1) * patch_h] = 1

            unlabeled_volume_batch_mix[b] = labeled_volume_batch[b] * img_mask + unlabeled_volume_batch[b] * (1 - img_mask)
            unlabel_pseudo_label_mix[b] = labeled_label[b] * img_mask + unlabel_pseudo_label[b] * (1 - img_mask)

            labeled_volume_batch_mix[b] = unlabeled_volume_batch[b] * img_mask_lab + labeled_volume_batch[b] * (1 - img_mask_lab)
            labeled_pseudo_label_mix[b] = unlabel_pseudo_label[b] * img_mask_lab + labeled_label[b] * (1 - img_mask_lab)

        volume_batch_mix = torch.cat([labeled_volume_batch_mix, unlabeled_volume_batch_mix], dim=0)
        label_batch_mix = torch.cat([labeled_pseudo_label_mix, unlabel_pseudo_label_mix], dim=0)

        pred_UNet_mix, pred_VNet_mix, pred_UNet_soft_mix, pred_VNet_soft_mix, fusion_map_mix = self.SGDL(volume_batch_mix)

        pseudo_label_mix = torch.argmax(fusion_map_mix, dim=1)

        fusion_map_soft_mix = torch.softmax(fusion_map_mix, dim=1)
        UNet_sup_mixed_loss = ce_loss(pred_UNet_mix, label_batch_mix.long()) + self.dice_loss(pred_UNet_soft_mix, label_batch_mix)
        UNet_enp_mixed_loss = self.entropy_loss(pred_UNet_soft_mix, C=2)
        UNet_cons_mixed_loss = loss_diff1(pred_UNet_soft_mix, pred_VNet_soft_mix.clone().detach())
        UNet_unsup_mixed_loss = ce_loss(pred_UNet_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:].long()) + self.dice_loss(pred_UNet_soft_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:])

        VNet_sup_mixed_loss = ce_loss(pred_VNet_mix, label_batch_mix.long()) + self.dice_loss(pred_VNet_soft_mix, label_batch_mix)
        VNet_enp_mixed_loss = self.entropy_loss(pred_VNet_soft_mix, C=2)
        VNet_cons_mixed_loss = loss_diff2(pred_VNet_soft_mix, pred_UNet_soft_mix.clone().detach())
        VNet_unsup_mixed_loss = ce_loss(pred_VNet_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:].long()) + self.dice_loss(pred_VNet_soft_mix[self.args.labeled_bs:], pseudo_label_mix[self.args.labeled_bs:])

        fusion_mixed_loss = ce_loss(fusion_map_mix, label_batch_mix.long()) + self.dice_loss(fusion_map_soft_mix, label_batch_mix)

        UNet_mixed_loss = UNet_sup_mixed_loss + 0.9 * UNet_enp_mixed_loss + consistency_weight * (UNet_cons_mixed_loss + UNet_unsup_mixed_loss)
        VNet_mixed_loss = VNet_sup_mixed_loss + 0.9 * VNet_enp_mixed_loss + consistency_weight * (VNet_cons_mixed_loss + VNet_unsup_mixed_loss)

        return UNet_mixed_loss, VNet_mixed_loss, fusion_mixed_loss

    def train(self, volume_batch, label_batch, iter_num):
        image_embeddings = self.sam_model.image_encoder(volume_batch)
        pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(volume_batch)

        fusion_map_soft = torch.softmax(fusion_map, dim=1)
        points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)
        low_res_masks_all = torch.empty((self.args.batch_size, 0, int(self.args.image_size/4), int(self.args.image_size/4)), device=self.args.device)

        for i in range(self.args.num_classes):
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                # points=points_embedding[i].unsqueeze(0),
                points=None,
                boxes=boxes_embedding[i],
                # boxes=None,
                masks=F.interpolate(fusion_map[:, i, ...].unsqueeze(1).clone().detach(), size=(64, 64), mode='bilinear')
                # masks=None,
            )

            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.args.multimask,
            )

            low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)

        pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size), mode="bilinear", align_corners=False)
        pred_sam_soft = torch.softmax(pred_sam, dim=1)

        fusion_loss = ce_loss(fusion_map[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(fusion_map_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        
        UNet_sup_loss = ce_loss(pred_UNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_UNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        UNet_cons_loss = loss_diff1(pred_UNet_soft, pred_VNet_soft.clone().detach())
        UNet_enp_loss = self.entropy_loss(pred_UNet_soft, C=2)
        UNet_kd_loss = self.KDLoss(pred_UNet.permute(0, 2, 3, 1).reshape(-1, 2), pred_sam.clone().detach().permute(0, 2, 3, 1).reshape(-1, 2))

        VNet_sup_loss = ce_loss(pred_VNet[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_VNet_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        VNet_cons_loss = loss_diff2(pred_VNet_soft, pred_UNet_soft.clone().detach())
        VNet_enp_loss = self.entropy_loss(pred_VNet_soft, C=2)
        VNet_kd_loss = self.KDLoss(pred_VNet.permute(0, 2, 3, 1).reshape(-1, 2), pred_sam.clone().detach().permute(0, 2, 3, 1).reshape(-1, 2))

        sam_sup_loss = ce_loss(pred_sam[:self.args.labeled_bs], label_batch[:self.args.labeled_bs].long()) + self.dice_loss(pred_sam_soft[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])

        consistency_weight = self.get_current_consistency_weight(iter_num // int(self.args.max_iterations/self.args.consistency_rampup)) * 10

        UNet_loss = UNet_sup_loss + UNet_kd_loss + 0.9 * UNet_enp_loss + consistency_weight * UNet_cons_loss
        VNet_loss = VNet_sup_loss + VNet_kd_loss + 0.9 * VNet_enp_loss + consistency_weight * VNet_cons_loss

        if iter_num > self.args.mixed_iterations:
            UNet_sup_mixed_loss, VNet_sup_mixed_loss, fusion_mixed_loss = self.mix_up(fusion_map_soft, volume_batch, pred_sam_soft[self.args.labeled_bs:], label_batch[:self.args.labeled_bs], consistency_weight)
            SGDL_loss = (UNet_loss + UNet_sup_mixed_loss + VNet_loss + VNet_sup_mixed_loss) / 2 + fusion_loss + fusion_mixed_loss
        else:
            SGDL_loss = (UNet_loss + VNet_loss) / 2 + fusion_loss

        sam_loss = sam_sup_loss

        self.optimizer_sam.zero_grad()
        self.optimizer_SGDL.zero_grad()

        sam_loss.backward()
        SGDL_loss.backward()

        self.optimizer_sam.step()
        self.optimizer_SGDL.step()

        lr_ = self.args.lr * (1.0 - iter_num / self.args.max_iterations)
        UNet_lr_ = self.args.UNet_lr * (1.0 - iter_num / self.args.max_iterations)

        for param_group in self.optimizer_sam.param_groups:
            param_group['lr'] = lr_
        for param_group in self.optimizer_SGDL.param_groups:
            param_group['lr'] = UNet_lr_

        logging.info('iteration %d : '
                     '  sam_loss : %f'
                     '  sam_lr_ : %10f'
                     
                     '  SGDL_loss : %f'
                     '  UNet_VNet_loss : %f'
                     '  fusion_loss : %f'
                     '  UNet_lr_ : %10f'

                     % (iter_num, sam_loss.item(), lr_,
                        SGDL_loss.item(), (UNet_loss + VNet_loss) / 2, fusion_loss,  UNet_lr_,
                        ))

    def val(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()

        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0
        avg_dice_unet = 0.0
        avg_dice_vnet = 0.0

        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image_embeddings = self.sam_model.image_encoder(val_image)
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = self.SGDL(val_image)

            points_embedding, boxes_embedding, mask_embedding = self.sam_model.super_prompt(image_embeddings)

            low_res_masks_all = torch.empty(
                (1, 0, int(self.args.image_size / 4), int(self.args.image_size / 4)),
                device=self.args.device)
            with torch.no_grad():
                for i in range(self.args.num_classes):
                    sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                        points=None,
                        boxes=boxes_embedding[i],
                        masks=F.interpolate(fusion_map[:, i, ...].unsqueeze(1).clone().detach(), size=(64, 64), mode='bilinear')
                    )
                    low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=self.args.multimask,
                    )
                    low_res_masks_all = torch.cat((low_res_masks_all, low_res_masks), dim=1)
            pred_sam = F.interpolate(low_res_masks_all, size=(self.args.image_size, self.args.image_size))
            pred_sam_soft = torch.softmax(pred_sam, dim=1)
            dice_sam = dice_coef(val_label, pred_sam_soft, thr=0.5)
            avg_dice_sam += dice_sam

            fusion_map_soft = torch.softmax(fusion_map, dim=1)
            dice_SGDL = dice_coef(val_label, fusion_map_soft, thr=0.5)
            avg_dice_SGDL += dice_SGDL

            dice_unet = dice_coef(val_label, pred_UNet_soft, thr=0.5)
            avg_dice_unet += dice_unet
            dice_vnet = dice_coef(val_label, pred_VNet_soft, thr=0.5)
            avg_dice_vnet += dice_vnet

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_SGDL = avg_dice_SGDL / len(val_loader)
        avg_dice_unet = avg_dice_unet / len(val_loader)
        avg_dice_vnet = avg_dice_vnet / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  SGDL_mean_dice : %f '
                     '  unet_mean_dice : %f '
                     '  vnet_mean_dice : %f '
                    % (iter_num, avg_dice_sam, avg_dice_SGDL, avg_dice_unet, avg_dice_vnet))

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)

        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model.pth')
            # save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model_' + str(iter_num) + '.pth')
            torch.save(self.SGDL.state_dict(), save_best_SGDL)
        self.sam_model.train()
        self.SGDL.train()

    def val_ACDC(self, val_loader, snapshot_path, iter_num):
        self.sam_model.eval()
        self.SGDL.eval()

        avg_dice_sam = 0.0
        avg_dice_SGDL = 0.0

        sam_info = np.array([0, 0, 0]).astype("float32")
        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            metric_list = test_single_volume(self.args, val_image, val_label, self.sam_model, self.SGDL)
            metric_list = np.array(metric_list).astype("float32")

            sam_info += metric_list[:, 0]

            metric_list = np.mean(metric_list, axis=0)
            avg_dice_sam += metric_list[0]
            avg_dice_SGDL += metric_list[1]

        avg_dice_sam = avg_dice_sam / len(val_loader)
        avg_dice_SGDL = avg_dice_SGDL / len(val_loader)

        sam_info = sam_info / len(val_loader)

        logging.info('iteration %d : '
                     '  sam_mean_dice : %f '
                     '  SGDL_mean_dice : %f '
                     '  sam_info : \n%s '
                     % (iter_num, avg_dice_sam, avg_dice_SGDL, str(sam_info)))

        if avg_dice_sam > self.best_performance_sam:
            self.best_performance_sam = avg_dice_sam
            save_best_sam = os.path.join(snapshot_path, 'sam_best_model.pth')
            torch.save(self.sam_model.state_dict(), save_best_sam)
        if avg_dice_SGDL > self.best_performance_SGDL:
            self.best_performance_SGDL = avg_dice_SGDL
            save_best_SGDL = os.path.join(snapshot_path, 'SGDL_best_model.pth')
            # save_best_SGDL = os.path.join(snapshot_path, 'SGDL_iter_' + str(iter_num) + ".pth")
            torch.save(self.SGDL.state_dict(), save_best_SGDL)

        self.sam_model.train()
        self.SGDL.train()