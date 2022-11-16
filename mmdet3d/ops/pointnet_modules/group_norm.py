# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmcv.ops import ball_query, grouping_operation, knn


class QueryAndGroupNorm(nn.Module):
    """Groups points with a ball query of radius.

    Args:
        max_radius (float): The maximum radius of the balls.
            If None is given, we will use kNN sampling instead of ball query.
        sample_num (int): Maximum number of features to gather in the ball.
        min_radius (float, optional): The minimum radius of the balls.
            Default: 0.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        return_grouped_xyz (bool, optional): Whether to return grouped xyz.
            Default: False.
        normalize_xyz (bool, optional): Whether to normalize xyz.
            Default: False.
        uniform_sample (bool, optional): Whether to sample uniformly.
            Default: False
        return_unique_cnt (bool, optional): Whether to return the count of
            unique samples. Default: False.
        return_grouped_idx (bool, optional): Whether to return grouped idx.
            Default: False.
    """

    def __init__(self,
                 max_radius,
                 sample_num,
                 channel,
                 min_radius=0,
                 use_xyz=True,
                 return_grouped_xyz=False,
                 normalize_xyz=False,
                 uniform_sample=False,
                 return_unique_cnt=False,
                 return_grouped_idx=False):
        super().__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        self.return_grouped_idx = return_grouped_idx
        if self.return_unique_cnt:
            assert self.uniform_sample, \
                'uniform_sample should be True when ' \
                'returning the count of unique samples'
        if self.max_radius is None:
            assert not self.normalize_xyz, \
                'can not normalize grouped xyz when max_radius is None'
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + 3]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + 3]))

    def forward(self, points_xyz, center_xyz, center_feature, features=None):
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the
                points.
            center_xyz (torch.Tensor): (B, npoint, 3) coordinates of the
                centriods.
            features (torch.Tensor): (B, C, N) The features of grouped
                points.

        Returns:
            torch.Tensor: (B, 3 + C, npoint, sample_num) Grouped
            concatenated coordinates and features of points.
        """
        # if self.max_radius is None, we will perform kNN instead of ball query
        # idx is of shape [B, npoint, sample_num]
        if self.max_radius is None:
            idx = knn(self.sample_num, points_xyz, center_xyz, False)
            idx = idx.transpose(1, 2).contiguous()
        else:
            idx = ball_query(self.min_radius, self.max_radius, self.sample_num,
                             points_xyz, center_xyz)

        if self.uniform_sample:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0,
                        num_unique, (self.sample_num - num_unique, ),
                        dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        # (B, 3, npoint, sample_num)
        grouped_xyz = grouping_operation(xyz_trans, idx)
        
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                # (B, C + 3, npoint, sample_num)--->(B, npoint, sample_num, C+3) 
                norm_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1).permute(0, 2, 3, 1).contiguous()
                # feature affine module
                mean = torch.cat([center_xyz, center_feature], dim=-1)
                # mean = torch.mean(norm_features, dim=2, keepdim=True)
                std = torch.std((norm_features-mean).reshape(norm_features.shape[0], -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
                norm_features = (norm_features-mean)/(std + 1e-5)
                norm_features = self.affine_alpha * norm_features + self.affine_beta
                norm_features = norm_features.permute(0, 3, 1, 2).contiguous()
                new_features = norm_features
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz
                    ), 'Cannot have not features and not use xyz as a feature!'
            new_features = None

        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_grouped_idx:
            ret.append(idx)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

