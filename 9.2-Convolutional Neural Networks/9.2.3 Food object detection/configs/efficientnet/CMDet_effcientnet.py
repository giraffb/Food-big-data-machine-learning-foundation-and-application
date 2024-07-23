_base_ = [
    './custom_retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

image_size = (608, 480)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa

# checkpoint = '/home/stu_6/NLM/myProject/mmdetection-main/work_dirs/CMDet_efficient_ft/epoch_17.pth'  # noqa
model = dict(
    data_preprocessor=dict(
        type='customDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,

        batch_augments=batch_augments
    ),
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=False, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    backbone_d=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=False, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    neck_d=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),

    # neck_d=dict(
    #     type='DilatedEncoder',
    #     in_channels=384,
    #     out_channels=256,
    #     block_mid_channels=128,
    #     num_residual_blocks=4,
    #     block_dilations=[2, 4, 6, 8]),
    neck_f=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    bbox_head_d=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # bbox_head_d=dict(
    #     type='YOLOFHead',
    #
    #     num_classes=228,
    #     in_channels=256,
    #     # in_channels=256,
    #
    #     reg_decoded_bbox=True,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         ratios=[1.0],
    #         scales=[1, 2, 4, 8, 16],
    #         strides=[32]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[.0, .0, .0, .0],
    #         target_stds=[1., 1., 1., 1.],
    #         add_ctr_clamp=True,
    #         ctr_clamp=32),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),

    bbox_head_f=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),

    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

# dataset settings
train_pipeline = [
    dict(type='customLoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.8, 1.2),
    #     keep_ratio=True),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    # dict(type='RandomCrop', crop_size=image_size),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='customPackDetInputs',
        meta_keys=('img_id', 'img_path', 'img_path_d', 'ori_shape', 'img_shape',
                   'scale_factor')
    )
]
test_pipeline = [
    dict(type='customLoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='customPackDetInputs',
        meta_keys=('img_id', 'img_path', 'img_path_d', 'ori_shape', 'img_shape',
                   'scale_factor')
    )
]
train_dataloader = dict(
    batch_size=4, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.04),
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))

# learning policy
max_epochs = 50
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)

# cudnn_benchmark=True can accelerate fix-size training
env_cfg = dict(cudnn_benchmark=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
work_dir = './work_dirs/CMDet_efficientnet_EMA_no_depth_branch'
