import numpy as np
# import spconv
import spconv.pytorch as spconv
import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

### us start
class LVA(nn.Module):
    def __init__(self):
        super(LVA, self).__init__()
        self.linear = spconv.SparseSequential(
            nn.Linear(192, 192)
        )
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.linear(x)
### us end

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")

        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None, fusion=False):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img=None):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + 'up3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 cfgs,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_seg_features=256,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)

        self.sparse_shape = sparse_shape

        #####
        self.down_seg = nn.Sequential(
            nn.Linear(num_seg_features, init_size * 2)  # us
        )
        self.down_pix = nn.Sequential(
            nn.Linear(num_seg_features, init_size * 2)  # us
        )
        #####
        self.LVABlock = LVA() # us

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")

        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")
        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")
        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

        self.upBlock3_ins_heatmap = UpBlock(4 * init_size, 2 * init_size, indice_key="up3_ins_heatmap", up_key="down2")
        self.upBlock3_ins_offset = UpBlock(4 * init_size, 2 * init_size, indice_key="up3_ins_offset", up_key="down2")
        self.upBlock3_ins_instmap = UpBlock(4 * init_size, 2 * init_size, indice_key="up3_ins_instmap", up_key="down2")
        self.ReconNet_ins_heatmap = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon_ins_heatmap")
        self.ReconNet_ins_offset = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon_ins_offset")
        self.ReconNet_ins_instmap = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon_ins_instmap")
        self.compress_offset = spconv.SubMConv3d(4 * init_size, 32, indice_key="compress_offset", kernel_size=3,
                                                 stride=1, padding=1, bias=True)
        self.compress_instmap = spconv.SubMConv3d(4 * init_size, 32, indice_key="compress_instmap", kernel_size=3,
                                                  stride=1, padding=1, bias=True)

        self.pool3d_heatmap = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_heatmap")
        self.pool3d_offset = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_offset")
        self.pool3d_instmap = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_instmap")
        self.logits_offset = nn.Conv2d(32, 2, 3, padding=(1, 0))
        self.logits_instmap = nn.Conv2d(32, 2, 3, padding=(1, 0))

        ###
        # concat (2*in , 2*in , 2*in)
        self.seconBlock0 = ResBlock(2 * 3 * init_size, 4 * 3 * init_size, 0.2, height_pooling=True, indice_key="sdown0")
        self.seconBlock1 = ResBlock(4 * 3 * init_size, 8 * 3 * init_size, 0.2, pooling=True, height_pooling=False,
                                    indice_key="sdown1")
        # self.seconBlock0 = ResBlock(2 * 2 * init_size, 4 * 2 *  init_size, 0.2, height_pooling=True, indice_key="sdown0")
        # self.seconBlock1 = ResBlock(4 * 2 *  init_size, 8 * 2 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="sdown1")

        self.seconup0 = UpBlock(8 * 3 * init_size, 8 * 3 * init_size, indice_key="sup0", up_key="sdown1")
        self.seconup1 = UpBlock(8 * 3 * init_size, 4 * 3 * init_size, indice_key="sup1", up_key="sdown0")
        # self.seconup0 = UpBlock(8 * 2 * init_size, 8 * 2 * init_size, indice_key="sup0", up_key="sdown1")
        # self.seconup1 = UpBlock(8 * 2 * init_size, 4 * 2 * init_size, indice_key="sup1", up_key="sdown0")

        # self.secon_compress0 = spconv.SubMConv3d(4 * 2 * init_size, 2 * init_size, indice_key="second_compress", kernel_size=3,
        #                                          stride=1, padding=1,bias=True)
        # self.secon_compress1 = spconv.SubMConv3d(2 * init_size, 64, indice_key="second_compress", kernel_size=3,
        #                                          stride=1, padding=1,bias=True)

        self.secon_compress0 = spconv.SubMConv3d(2 * 4 * init_size, 2 * 2 * init_size, indice_key="second_compress",
                                                 kernel_size=3,
                                                 stride=1, padding=1, bias=True)
        self.secon_compress1 = spconv.SubMConv3d(6 * init_size, 2 * init_size, indice_key="second_compress",
                                                 kernel_size=3,
                                                 stride=1, padding=1, bias=True)

        self.resSeg = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down_seg2")
        self.resPix = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down_pix2")
        self.downCntxSeg = ResContextBlock(init_size * 2, init_size * 2, indice_key="pre1")  # us
        self.downCntxPix = ResContextBlock(init_size * 2, init_size * 2, indice_key="pre2")  # us
        #####

    def forward(self, voxel_features, coors, batch_size, pooled):
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)

        ###
        feat_seg = self.down_seg(pooled[0])
        feat_pix = self.down_pix(pooled[1])

        ret_seg = spconv.SparseConvTensor(feat_seg, coors, self.sparse_shape, batch_size)
        ret_pix = spconv.SparseConvTensor(feat_pix, coors, self.sparse_shape, batch_size)
        ###
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)  # 64
        down2c, down2b = self.resBlock3(down1c)  # 128
        down3c, down3b = self.resBlock4(down2c)  # 256
        down4c, down4b = self.resBlock5(down3c)  # 512

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)  # [1, 64, 480, 360, 32]

        # breakpoint()
        ####
        ret_seg = self.downCntxSeg(ret_seg)  # [1, 64, 480, 360, 32]
        ret_pix = self.downCntxPix(ret_pix)  # [1, 64, 480, 360, 32]
        up1e_res = up1e  # us

        up1e = up1e.replace_feature(
            torch.cat((up1e.features, ret_seg.features, ret_pix.features), 1))  # [1, 192, 480, 360, 32]
        up1e = self.LVABlock(up1e)  # us

        scomp0 = self.secon_compress1(up1e)  # [1, 64, 480, 360, 32]
        scomp0 = scomp0.replace_feature(scomp0.features + up1e_res.features)  # us

        #####
        up0e = self.ReconNet(scomp0)
        up0e = up0e.replace_feature(torch.cat((up0e.features, scomp0.features), 1))
        logits = self.logits(up0e)
        logits = logits.dense()

        up0e_ins_heatmap = self.ReconNet_ins_heatmap(scomp0)
        up0e_ins_heatmap = up0e_ins_heatmap.replace_feature(
            torch.cat((up0e_ins_heatmap.features, scomp0.features), 1))
        up0e_ins_heatmap = self.pool3d_heatmap(up0e_ins_heatmap)  # [1, 128, 480, 360, 1]
        heatmap = up0e_ins_heatmap.dense().squeeze(-1)

        up0e_ins_offset = self.ReconNet_ins_offset(scomp0)
        up0e_ins_offset = up0e_ins_offset.replace_feature(
            torch.cat((up0e_ins_offset.features, scomp0.features), 1))
        up0e_ins_offset = self.pool3d_offset(up0e_ins_offset)  # [1, 128, 480, 360, 1]
        up0e_ins_offset = self.compress_offset(up0e_ins_offset)  # [1, 32, 480, 360, 1]
        offset = up0e_ins_offset.dense().squeeze(-1)  # [1, 32, 480, 360]
        offset = F.pad(offset, (1, 1, 0, 0), mode='circular')  # [1, 32, 480, 362]
        offset = self.logits_offset(offset)  # [1, 2, 480, 360]

        up0e_ins_instmap = self.ReconNet_ins_instmap(scomp0)
        up0e_ins_instmap = up0e_ins_instmap.replace_feature(
            torch.cat((up0e_ins_instmap.features, scomp0.features), 1))
        up0e_ins_instmap = self.pool3d_instmap(up0e_ins_instmap)
        up0e_ins_instmap = self.compress_instmap(up0e_ins_instmap)
        instmap = up0e_ins_instmap.dense().squeeze(-1)
        instmap = F.pad(instmap, (1, 1, 0, 0), mode='circular')
        instmap = self.logits_instmap(instmap)

        return logits, heatmap, offset, instmap, scomp0.features