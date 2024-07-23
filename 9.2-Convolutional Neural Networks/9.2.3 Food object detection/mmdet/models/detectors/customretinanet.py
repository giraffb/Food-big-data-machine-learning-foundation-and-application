# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
# from .single_stage import SingleStageDetector
from .custom_single_stage import customSingleStageDetector

@MODELS.register_module()
class customRetinaNet(customSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 backbone_d: ConfigType,
                 neck: ConfigType,
                 neck_d: ConfigType,
                 neck_f: ConfigType,
                 bbox_head: ConfigType,
                 bbox_head_d: ConfigType,
                 bbox_head_f: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            backbone_d=backbone_d,
            neck=neck,
            neck_d=neck_d,
            neck_f=neck_f,
            bbox_head=bbox_head,
            bbox_head_d=bbox_head_d,
            bbox_head_f=bbox_head_f,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
