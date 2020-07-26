"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict
import os
from concurrent.futures import ProcessPoolExecutor, wait

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from .resnet import resnet101


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {"align": False}

        # Encoder
        self.encoder = nn.Sequential(
            OrderedDict([("backbone", Encoder(in_channels, self.pretrained_path)),])
        )
        self.resNetEncoder = resnet101(pretrained=True)

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, saliency_pred):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [torch.cat(qry_imgs, dim=0),],
            dim=0,
        )
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[: n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size
        )  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size :].view(
            n_queries, batch_size, -1, *fts_size
        )  # N x B x C x H' x W'
        fore_mask = torch.stack(
            [torch.stack(way, dim=0) for way in fore_mask], dim=0
        )  # Wa x Sh x B x H' x W'
        back_mask = torch.stack(
            [torch.stack(way, dim=0) for way in back_mask], dim=0
        )  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        dist_loss = 0
        outputs = []
        for epi in range(batch_size):
            # ##### Compute the distance ######
            part = 3
            supp_fg_fts = [
                [
                    self.getFeatures_new(
                        supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]], part
                    )
                    for shot in range(n_shots)
                ]
                for way in range(n_ways)
            ]
            supp_bg_fts = [
                [
                    self.getFeatures_new(
                        supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]], part
                    )
                    for shot in range(n_shots)
                ]
                for way in range(n_ways)
            ]
            fg_prototypes, bg_prototypes = self.getPrototype_new(
                supp_fg_fts, supp_bg_fts
            )

            if not self.training:
                fg_prototypes, bg_prototypes = self.kmeans_adjustment(
                    epi,
                    n_shots,
                    n_ways,
                    supp_fts,
                    fore_mask,
                    back_mask,
                    fg_prototypes,
                    bg_prototypes,
                    part,
                )

            prototypes = bg_prototypes + fg_prototypes
            dist = [
                self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes
            ]

            # take max of each 9 dist results
            max_dist = []
            i = 0
            while i < len(dist):
                dists = torch.cat(dist[i : i + part * part], 0)
                max_dist.append(torch.max(dists, dim=0)[0].unsqueeze(0))
                i += part * part
            dist = max_dist
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            pred_full_size = F.interpolate(pred, size=img_size, mode="bilinear")
            outputs.append(pred_full_size)

            ###### Prototype alignment loss ######
            if self.config["align"] and self.training:
                align_loss_epi = self.alignLoss(
                    qry_fts[:, epi],
                    pred,
                    supp_fts[:, :, epi],
                    fore_mask[:, :, epi],
                    back_mask[:, :, epi],
                )
                align_loss += align_loss_epi

            # Distance Loss (use full prototype instead of multi prototype)
            supp_fg_fts = [
                [
                    self.getFeatures(
                        supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]]
                    )
                    for shot in range(n_shots)
                ]
                for way in range(n_ways)
            ]
            supp_bg_fts = [
                [
                    self.getFeatures(
                        supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]]
                    )
                    for shot in range(n_shots)
                ]
                for way in range(n_ways)
            ]

            fg_prototypes, bg_prototypes = self.getPrototype(supp_fg_fts, supp_bg_fts)
            for i in range(len(fg_prototypes)):
                dist_loss += 1 / torch.log(torch.dist(fg_prototypes[i], bg_prototypes))

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss, dist_loss

    def compute_min_distance(self, total_dist, indxes):
        lst = []
        for i in indxes:
            cur_min = float("inf")
            cuz_ind = -1
            for j, d in enumerate(total_dist):
                if d[i] < cur_min:
                    cur_min, cur_ind = d[i], j
            lst.append((cur_ind, cur_min))
        return lst

    def kmeans_adjustment(
        self,
        epi,
        n_shots,
        n_ways,
        supp_fts,
        fore_mask,
        back_mask,
        fg_prototypes,
        bg_prototypes,
        part,
    ):
        adjusted_fg_prototypes = []
        for i in range(1):
            for way in range(n_ways):
                ptps = fg_prototypes[way * part * part : (way + 1) * part * part]
                for ptp in ptps:
                    new_fg_ptp = []
                    for shot in range(n_shots):
                        new_fg_ptp.append(
                            self.kmeans_prototypes(
                                ptp,
                                supp_fts[way, shot, [epi]],
                                fore_mask[way, shot, [epi]],
                            )
                        )
                    adjusted_fg_prototypes.append(sum(new_fg_ptp) / n_shots)
        adjusted_bg_prototypes = []
        for ptp in bg_prototypes:
            new_bg_ptp = []
            for way in range(n_ways):
                for shot in range(n_shots):
                    new_bg_ptp.append(
                        self.kmeans_prototypes(
                            ptp, supp_fts[way, shot, [epi]], back_mask[way, shot, [epi]]
                        )
                    )
                adjusted_bg_prototypes.append(sum(new_bg_ptp) / (n_shots * n_ways))
        return adjusted_fg_prototypes, adjusted_bg_prototypes

    def kmeans_prototypes(self, prototype, fts, mask):
        """
        Perform approximate k-means to fit better prototype values
          k = num of prototypes
        For each class, compute distance between pixel to prototype
        The new prototype = sum(ft * similarity_measure) 
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")
        dist = F.cosine_similarity(
            fts * mask[None, ...], prototype[..., None, None], dim=1
        )
        weighted_prototype = torch.sum(dist * fts, dim=(2, 3)) / (
            torch.sum(dist) + 1e-5
        )  # (mask[None, ...].sum(dim=(2,3)) + 1e-5)
        return weighted_prototype

    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist

    def getFeatures_new(self, fts, mask, part=3):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")
        squared_h = int(mask.shape[1] / part)
        masked_fts = []
        for i in range(part):
            for j in range(part):
                if i == part - 1 and j == part - 1:
                    mask_p = mask[:, i * squared_h :, j * squared_h :]
                    ft_p = fts[:, :, i * squared_h :, j * squared_h :]
                elif j == part - 1:
                    mask_p = mask[
                        :, i * squared_h : (i + 1) * squared_h, j * squared_h :
                    ]
                    ft_p = fts[
                        :, :, i * squared_h : (i + 1) * squared_h, j * squared_h :
                    ]
                elif i == part - 1:
                    mask_p = mask[
                        :, i * squared_h :, j * squared_h : (j + 1) * squared_h
                    ]
                    ft_p = fts[
                        :, :, i * squared_h :, j * squared_h : (j + 1) * squared_h
                    ]
                else:
                    mask_p = mask[
                        :,
                        i * squared_h : (i + 1) * squared_h,
                        j * squared_h : (j + 1) * squared_h,
                    ]
                    ft_p = fts[
                        :,
                        :,
                        i * squared_h : (i + 1) * squared_h,
                        j * squared_h : (j + 1) * squared_h,
                    ]
                m_ft = torch.sum(ft_p * mask_p[None, ...], dim=(2, 3)) / (
                    mask_p[None, ...].sum(dim=(2, 3)) + 1e-5
                )  # 1 x C
                masked_fts.append(m_ft)

        return masked_fts

    def getPrototype_new(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        
        Args:
            Wa x Sh x num_split x [1 x C]
        """
        n_ways, n_shots, num_split = len(fg_fts), len(fg_fts[0]), len(fg_fts[0][0])
        fg_prototypes = []
        bg_prototypes = []

        for i in range(num_split):
            for way in fg_fts:
                split_fg = []
                for shot in way:
                    split_fg.append(shot[i])
                fg_prototypes.append(sum(split_fg) / n_shots)

        for i in range(num_split):
            split_bg = []
            for way in bg_fts:
                for shot in way:
                    split_bg.append(shot[i])
            # sum each shot and way for each split
            bg_prototypes.append(sum(split_bg) / (n_shots * n_ways))

        return fg_prototypes, bg_prototypes

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (
            mask[None, ...].sum(dim=(2, 3)) + 1e-5
        )  # 1 x C
        return masked_fts

    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(
            binary_masks, dim=1
        ).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (
            pred_mask.sum((0, 3, 4)) + 1e-5
        )  # (1 + Wa) x C

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [
                    self.calDist(img_fts, prototype) for prototype in prototypes
                ]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(
                    supp_pred, size=fore_mask.shape[-2:], mode="bilinear"
                )
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=img_fts.device
                ).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = (
                    loss
                    + F.cross_entropy(
                        supp_pred, supp_label[None, ...], ignore_index=255
                    )
                    / n_shots
                    / n_ways
                )
        return loss

    def alignLoss_new(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch with multi prototypes

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(
            binary_masks, dim=1
        ).float()  # N x (1 + Wa) x 1 x H' x W'

        part = 3
        fg_mask = pred_mask[0, 1, :, :]
        bg_mask = pred_mask[0, 0, :, :]
        supp_fg_fts = [[self.getFeatures_new(qry_fts, fg_mask, part)]]
        supp_bg_fts = [[self.getFeatures_new(qry_fts, bg_mask, part)]]
        fg_prototypes, bg_prototypes = self.getPrototype_new(supp_fg_fts, supp_bg_fts)
        prototypes = fg_prototypes + bg_prototypes

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_dist = [
                    self.calDist(img_fts, prototype) for prototype in prototypes
                ]

                # take max of each 9 dist results
                max_dist = []
                i = 0
                while i < len(supp_dist):
                    dists = torch.cat(supp_dist[i : i + part * part], 0)
                    max_dist.append(torch.max(dists, dim=0)[0].unsqueeze(0))
                    i += part * part
                supp_dist = max_dist

                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(
                    supp_pred, size=fore_mask.shape[-2:], mode="bilinear"
                )
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=img_fts.device
                ).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = (
                    loss
                    + F.cross_entropy(
                        supp_pred, supp_label[None, ...], ignore_index=255
                    )
                    / n_shots
                    / n_ways
                )
        return loss

