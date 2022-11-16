# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet3d.ops import PAConv
from torch import nn as nn
from torch.nn import functional as F

from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points

from .builder import SA_MODULES
from .group_norm import QueryAndGroupNorm


class BasePointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional):
            Range of points to apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        grouper_return_grouped_xyz (bool, optional): Whether to return
            grouped xyz in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool, optional): Whether to return
            grouped idx in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 group_norm=False,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(BasePointSAModule, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                self.group_norm = group_norm
                if self.group_norm:
                    grouper = QueryAndGroupNorm(
                        radius,
                        sample_num,
                        channel=mlp_channels[0][0],
                        min_radius=min_radius,
                        use_xyz=use_xyz,
                        normalize_xyz=normalize_xyz,
                        return_grouped_xyz=grouper_return_grouped_xyz,
                        return_grouped_idx=grouper_return_grouped_idx)
                else:
                    grouper = QueryAndGroup(
                        radius,
                        sample_num,
                        min_radius=min_radius,
                        use_xyz=use_xyz,
                        normalize_xyz=normalize_xyz,
                        return_grouped_xyz=grouper_return_grouped_xyz,
                        return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _sample_points(self, points_xyz, features, indices, target_xyz):
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
            new_feature = gather_points(features, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
            # TODO: indices--->features--->new_features
            new_feature = None
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
                new_feature = gather_points(features,
                                         indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None
                new_feature = None

        return new_xyz, new_feature, indices

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(
        self,
        points_xyz,
        features=None,
        indices=None,
        target_xyz=None,
    ):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) features of each point.
                Default: None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor, optional): (B, M, 3) new coords of the outputs.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, new_feature, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)

        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            if self.group_norm:
                grouped_results = self.groupers[i](points_xyz, new_xyz, new_feature, features)
            else:
                grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            # (B, mlp[-1], num_point, nsample)
            
            new_features = self.mlps[i](grouped_results)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices


@SA_MODULES.register_module()
class PointSAModuleMSG(BasePointSAModule):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional): Range of points to
            apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 group_norm=False,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(PointSAModuleMSG, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            fps_mod=fps_mod,
            group_norm=group_norm,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            if use_xyz:
                if self.group_norm:
                    mlp_channel[0] = mlp_channel[0] + 3
                else:
                    mlp_channel[0] += 3
            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)


@SA_MODULES.register_module()
class PointSAModule(PointSAModuleMSG):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int, optional): Number of points.
            Default: None.
        radius (float, optional): Radius to group with.
            Default: None.
        num_sample (int, optional): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int], optional): Range of points
            to apply FPS. Default: [-1].
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 group_norm=False,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False):
        super(PointSAModule, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            group_norm=group_norm,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)



@SA_MODULES.register_module()
class Local_to_global_reason(PointSAModuleMSG):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional): Range of points to
            apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 mlp_channels,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 group_norm=False,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False):
        super(Local_to_global_reason, self).__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            group_norm=group_norm,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)

        # view self.mlps as local extractor
        self.global_mlps = nn.ModuleList()
        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            # if use_xyz:
            #     if self.group_norm:
            #         mlp_channel[0] = mlp_channel[0] - 3
            #     else:
            #         mlp_channel[0] -= 3
            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias='auto'))
            self.global_mlps.append(mlp)
        
        # fuse local-to-global feature
        # self.conv = nn.Conv2d(self.mlp_channels[0][-1] * 2, self.mlp_channels[0][-1], 1, 1)
        self.conv = ConvModule(self.mlp_channels[0][-1] * 2,
                               self.mlp_channels[0][-1],
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               norm_cfg=norm_cfg,
                               act_cfg=None)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SELayer(self.mlp_channels[0][-1])
        # self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        points_xyz,
        features=None,
        indices=None,
        target_xyz=None,
    ):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) features of each point.
                Default: None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor, optional): (B, M, 3) new coords of the outputs.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, new_feature, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)

        global_feature = torch.cat([new_xyz, new_feature], dim=-1).transpose(1, 2).unsqueeze(dim=-1)
        # global_feature = new_feature.transpose(1, 2).unsqueeze(dim=-1)
        
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            if self.group_norm:
                grouped_results = self.groupers[i](points_xyz, new_xyz, new_feature, features)
            else:
                grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)
            # view new_features as local features
            global_feature = self.global_mlps[i](global_feature).repeat(1, 1, 1, new_features.size()[-1])
            
            fused_features = torch.cat([new_features, global_feature], dim=1)
            
            # fused_features = self.softmax(self.conv(fused_features))
            # new_features = fused_features[:, :new_features.size()[1], :, :] * new_features +\
            #     fused_features[:, new_features.size()[1]::, :, :] * global_feature

            # import pdb; pdb.set_trace()
            new_features = self.relu(self.attention(self.conv(fused_features)))

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), indices


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)