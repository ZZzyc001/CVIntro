from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet
# ----------TODO------------


class PointNetfeat(nn.Module):

    def __init__(self, global_feat=True, d=1024):
        super(PointNetfeat, self).__init__()

        self.d = d
        self.global_feat = global_feat
        self.first_layer = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.second_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.d),
            nn.BatchNorm1d(self.d),
            nn.ReLU(),
        )

    def forward(self, x):
        batchsize, n_pts, _ = x.size()

        x = x.reshape(-1, 3)
        feature = self.first_layer(x)
        up_feature = self.second_layer(feature)
        if self.global_feat:
            up_feature = up_feature.reshape(batchsize, n_pts,
                                            -1).max(1, keepdim=True)[0]
            up_feature = up_feature.repeat(1, n_pts, 1)

            feature = torch.cat(
                [feature.reshape(batchsize, n_pts, -1), up_feature],
                2).reshape(batchsize * n_pts, -1)
            return feature
        else:
            vis_feature = up_feature.reshape(batchsize, n_pts, -1)
            up_feature = up_feature.reshape(batchsize, n_pts, -1).max(1)[0]
            return up_feature, vis_feature


class PointNetCls1024D(nn.Module):

    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()

        self.k = k
        self.feat_layer = PointNetfeat(global_feat=False, d=1024)
        self.out_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.k),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):

        x, vis_feature = self.feat_layer(x)
        x = self.out_layer(x)

        return x, vis_feature  # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):

    def __init__(self, k=2):
        super(PointNetCls256D, self).__init__()

        self.k = k
        self.feat_layer = PointNetfeat(global_feat=False, d=256)
        self.out_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.k),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):

        x, vis_feature = self.feat_layer(x)
        x = self.out_layer(x)

        return x, vis_feature


class PointNetSeg(nn.Module):

    def __init__(self, k=2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feat_layer = PointNetfeat(global_feat=True, d=1024)
        self.out_layer = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.k),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]

        x = self.feat_layer(x)
        x = self.out_layer(x).reshape(batchsize, n_pts, -1)
        x = x.contiguous()

        return x
