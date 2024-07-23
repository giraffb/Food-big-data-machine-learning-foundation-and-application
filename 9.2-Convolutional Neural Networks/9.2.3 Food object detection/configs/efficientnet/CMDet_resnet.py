_base_ = [
    # './custom_retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

image_size = (1333, 800)
# batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
norm_cfg = dict(type='BN', requires_grad=True)
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa

model = dict(
    type='customRetinaNet',
    data_preprocessor=dict(
        type='customDetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        # pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    backbone_d=dict(
        type='ResNet',
        # depth=18,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,
        relu_before_extra_convs = True,
        no_norm_on_lateral = True,
        norm_cfg=norm_cfg
    ),
    neck_d=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg
    ),
    neck_f=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        # in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=228,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    bbox_head_d=dict(
        type='RetinaHead',
        num_classes=228,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    bbox_head_f=dict(
        type='RetinaHead',
        num_classes=228,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# dataset settings
train_pipeline = [
    dict(type='customLoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomShift', prob=0.5, max_shift_px=32),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'img_path_d', 'ori_shape', 'img_shape',
                    'scale_factor')
         )
]
test_pipeline = [
    dict(type='customLoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img_path_d','ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))

# learning policy
max_epochs = 100
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
