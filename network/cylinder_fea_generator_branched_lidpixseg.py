# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_scatter

from network.swiftnet import SwiftNetRes18
from network.util.LIfusion_block import Feature_Gather, Atten_Fusion_Conv
from network.util.cam import returnCAM
from network.pvp_generation import shift_voxel_grids, return_tensor_index, return_tensor_index_v2

INDEX_SHIFT = [[0,-1,-1,-1], [0, -1,-1,0], [0, -1,-1,1], [0, -1,0,-1], [0, -1,0,0], [0,-1,0,1],
               [0,-1,1,-1], [0,-1,1,0], [0,-1,1,1], [0,0,-1,-1], [0,0,-1,0], [0,0,-1,1],
               [0,0,0,-1], [0,0,0,0], [0,0,0,1], [0,0,1,-1], [0,0,1,0], [0,0,1,1],
               [0,1,-1,-1],[0,1,-1,0], [0,1,-1,1], [0,1,0,-1], [0,1,0,0], [0,1,0,1], 
               [0,1,1,-1], [0,1,1,0], [0,1,1,1]]

class cylinder_fea(nn.Module):

    def __init__(self, cfgs, grid_size, nclasses, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None, use_sara=False, tau=0.7, 
                 use_att=False, head_num=2):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.nclasses = nclasses
        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        self.use_pix_fusion = cfgs['model']['pix_fusion']
        self.to_128 = nn.Sequential(
                nn.Linear(self.pool_dim, 128)
            )

        if self.fea_compre is not None:
            if self.use_pix_fusion:
                self.fea_compression = nn.Sequential(
                    nn.Linear(128, self.fea_compre),
                    nn.ReLU())
            else:
                self.fea_compression = nn.Sequential(
                    nn.Linear(128, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

        # gh
        # TODO add an option for using both seg and pix fea e.i, 256 --> 512
        self.segfea_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.pixfea_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )


    def forward(self, pt_fea, xy_ind, fusion_dict):
        cur_dev = pt_fea[0].get_device()
        pt_ind = []
        for i_batch in range(len(xy_ind)):
            pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        # MLP sub-branch
        mlp_fea = []
        for _, f in enumerate(pt_fea):
            mlp_fea.append(self.PPmodel(f)) # MLP Features

        # gh; SAM features
        segfea_tensor, pixfea_tensor = fusion_dict['segfea'], fusion_dict['pixfea']
        cat_sam_fea = torch.cat(segfea_tensor, dim=0)  # [n_b*n_pts, 256] = [490769, 256]
        cat_pix_fea = torch.cat(pixfea_tensor, dim=0)  # [n_b*n_pts, 256] = [490769, 256]

        # gh; point index
        cat_pt_ind = torch.cat(pt_ind, dim=0)   # [n_b*n_pts, 4] = [490769, 4]

        # point feature
        cat_mlp_fea = torch.cat(mlp_fea, dim=0)  # [n_b*n_pts, 256] = [490769, 256]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0) # ori_cylinder_data ↔ unq，unq_inv
        unq = unq.type(torch.int64) # [148957, 4]

        # get cylinder voxel features
        ori_cylinder_data = torch_scatter.scatter_max(cat_mlp_fea, unq_inv, dim=0)[0]
        ori_cylinder_data = self.to_128(ori_cylinder_data) # [148957, 128]


        """gs code start"""
        # sam feature
        seg_pooled = torch_scatter.scatter_max(cat_sam_fea, unq_inv, dim=0)[0]
        # zero/nonzero index
        device = seg_pooled.device
        nonzero_indices = torch.where(torch.any(seg_pooled != 0, dim=1))[0].to(device)
        zero_indices = torch.where(torch.all(seg_pooled == 0, dim=1))[0].to(device)
        # sam feature
        seg_pooled = self.segfea_to_128(seg_pooled) # [153791, 128]
        pix_pooled = torch_scatter.scatter_max(cat_pix_fea, unq_inv, dim=0)[0]
        pix_pooled = self.pixfea_to_128(pix_pooled) # [153791, 128]


        # get rows where seg feature row is zero / nonzero
        vox_nonzero = ori_cylinder_data[nonzero_indices] # [21469, 128]
        seg_nonzero = seg_pooled[nonzero_indices] # [21469, 128]
        pix_nonzero = pix_pooled[nonzero_indices] # [21469, 128]
        unq_nonzero = unq[nonzero_indices] # [21469, 128]
        vox_zero = ori_cylinder_data[zero_indices] # [132322, 128]


        # fuse none zero features
        cat_fea = torch.cat((vox_nonzero, seg_nonzero, pix_nonzero), dim=0)  # [21469 * 3, 128]
        cat_unq = torch.cat((unq_nonzero, unq_nonzero, unq_nonzero), dim=0)  # [21469 * 3, 4]
        unq_cat, unq_inv_cat, unq_cnt_cat = torch.unique(cat_unq, return_inverse=True, return_counts=True, dim=0)
        fused_cylinder_data = torch_scatter.scatter_add(cat_fea, unq_inv_cat, dim=0) # [21469, 128]

        # concatenate features back together
        output = torch.zeros_like(ori_cylinder_data).to(device) # (153791,128)
        output[nonzero_indices] = fused_cylinder_data
        output[zero_indices] = vox_zero

        # compression or not
        if self.fea_compre:
            processed_pooled_data = self.fea_compression(output) # (153791,16)
        else:
            processed_pooled_data = output
        """gs code end"""

        return unq, processed_pooled_data, None, None

