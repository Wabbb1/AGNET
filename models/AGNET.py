import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from viz_image_func import *
from .backbone import build_backbone
from .head import build_head
from .neck import build_neck
from .utils import Conv_BN_ReLU
from models.neck.visual_transformer_noxin import FilterBasedTokenizer, Transformer, Projector, Transformer_Decoder
import numpy as np
import cv2
from eval.ctw.eval import plot_heat

class AGNET(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(AGNET, self).__init__()
        self.backbone = build_backbone(backbone)
        self.WFEN = build_neck(neck)
        self.det_head = build_head(detection_head)
        self.flag = 1001


    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):
        outputs = dict()
        bs, ch, h, w = imgs.shape

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        f = self.WFEN(f[0], f[1], f[2], f[3])


        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)


        return outputs


if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    model = build_model(cfg.model)
