_base_ = [
    '../_base_/datasets/scannet-3d-18class.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        fps_mod_list=[['D-FPS'], ['F-FPS'], ['F-FPS'], ['F-FPS']],
        sa_channels=((128, 128, 256), (256, 256, 512), (256, 256, 512),
                     (256, 256, 512)),
        fp_channels=((512, 512), (512, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='Local_to_global_reason',
            pool_mod='max',
            group_norm=True,
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        num_classes=18,
        use_proposal=True,
        iou_loss=dict(
            type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 /
            3.0),
        vote_aggregation_cfg=dict(
                type='Local_to_global_reason',
                pool_mod='max',
                num_point=256,
                radius=0.3,
                num_sample=16,
                mlp_channels=[256, 128, 128, 128],
                group_norm=True,
                use_xyz=True,
                normalize_xyz=True),
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=18,
            num_dir_bins=1,
            with_rot=False,
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]])),
        train_cfg=dict(
            pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
        test_cfg=dict(
            sample_mod='seed',
            nms_thr=0.25,
            score_thr=0.02,
            per_class_proposal=True))

# yapf:disable
log_config = dict(interval=30)
# yapf:enable
