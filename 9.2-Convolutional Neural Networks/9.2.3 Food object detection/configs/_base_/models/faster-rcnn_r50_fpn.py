# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=228,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
#
# # 修改数据集相关配置
# data_root = 'data/coco/'
# metainfo = {
#     'classes':
#         ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#          'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#          'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#          'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
#          'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#          'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#          'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
#     # palette is a list of color tuples, which is used for visualization.
#     'palette':
#         [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
#          (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
#          (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
#          (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
#          (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
#          (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
#          (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
#          (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
#          (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
#          (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
#          (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
#          (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
#          (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
#          (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
#          (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
#          (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
#          (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
#          (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
#          (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
#          (246, 0, 122), (191, 162, 208)]
# }
# train_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/instances_train2017.json',
#         data_prefix=dict(img='annotations/')))
# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='annotations/')))
# test_dataloader = val_dataloader
#
# # 修改评价指标相关配置
# val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
# test_evaluator = val_evaluator


