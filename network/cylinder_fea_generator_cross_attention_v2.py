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
### gs start ###
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        self.drop_out = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x : [batch_size, 1, hidden_dim)
        x = self.drop_out(torch.relu(self.fc_1(x)))
        # x : [batch_size, 1, pf_dim]
        x = self.fc_2(x)
        # x : [batch_size, 1, hidden_dim)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio):
        super().__init__()
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, n_heads, dropout_ratio, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, src):
        # src : [batch_size, feat_dim, hidden_dim]
        # self attention
        _trg = self.self_attention(trg, src, src)[0]

        # dropout -> residual connection -> layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg : [batch_size, feat_dim, hidden_dim]

        # position-wise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout -> residual connection -> layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # src : [batch_size, feat_dim, hidden_dim]

        return trg

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(hidden_dim, n_heads, pf_dim, dropout_ratio) \
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, src):
        _trg = self.dropout(trg)
        _src = self.dropout(src)
        for layer in self.layers:
            _trg = layer(_trg, _src)
        return _trg
### gs end ###
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

        # gs
        self.segfea_1_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.segfea_2_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.segfea_3_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.pixfea_to_128 = nn.Sequential(
            nn.Linear(256, 128)
        )
        self.attention = CrossAttention(hidden_dim=256, n_layers=1, n_heads=1, pf_dim=512, dropout_ratio=0.1)
        self.comp_attn = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
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

        # fusion_tensor = torch.cat((pixfea_tensor, samfea_tensor), dim=1)  # todo both seg and pix fea
        cat_sam_fea = torch.cat(segfea_tensor, dim=0)  # [n_b*n_pts, 4]
        # gs
        cat_sam_fea_1 = cat_sam_fea[:,0,:]
        cat_sam_fea_2 = cat_sam_fea[:,1,:]
        cat_sam_fea_3 = cat_sam_fea[:,2,:]
        cat_pix_fea = torch.cat(pixfea_tensor, dim=0)

        # gh; feature fusion
        cat_pt_ind = torch.cat(pt_ind, dim=0)   # [n_b*n_pts, 4]

        # MLP sub-branch
        cat_mlp_fea = torch.cat(mlp_fea, dim=0)  # [n_b*n_pts, 256]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0) # ori_cylinder_data ↔ unq，unq_inv
        unq = unq.type(torch.int64)

        # get cylinder voxel features
        ori_cylinder_data = torch_scatter.scatter_max(cat_mlp_fea, unq_inv, dim=0)[0] #[150555, 256]


        # gs
        seg_pooled_1 = torch_scatter.scatter_max(cat_sam_fea_1, unq_inv, dim=0)[0] #[150555, 256]
        seg_pooled_2 = torch_scatter.scatter_max(cat_sam_fea_2, unq_inv, dim=0)[0]
        seg_pooled_3 = torch_scatter.scatter_max(cat_sam_fea_3, unq_inv, dim=0)[0]
        pix_pooled = torch_scatter.scatter_max(cat_pix_fea, unq_inv, dim=0)[0]
        pooled = [seg_pooled_1, seg_pooled_2, seg_pooled_3, pix_pooled]

        """ gs attention start """
        trg_fea = ori_cylinder_data[:, None, :]  # [131515, 1, 256]
        src_fea = torch.cat((pix_pooled[:, None, :], seg_pooled_1[:, None, :], seg_pooled_2[:, None, :], seg_pooled_3[:, None, :]), 1)  # [127899, 4, 256]
        attn_fea = self.attention(trg_fea, src_fea)  # [127899, 1, 256]
        cat_attn_fea = torch.cat((trg_fea, attn_fea), 2) # [127899, 1, 512]
        comp_attn_fea = self.comp_attn(cat_attn_fea) # [127899, 1, 128]
        fused_cylinder_data = comp_attn_fea.squeeze(1)  # [127899, 128]
        """ gs attention end """


        if self.fea_compre:
            processed_pooled_data = self.fea_compression(fused_cylinder_data)
        else:
            processed_pooled_data = fused_cylinder_data
        return unq, processed_pooled_data, pooled, unq_inv
