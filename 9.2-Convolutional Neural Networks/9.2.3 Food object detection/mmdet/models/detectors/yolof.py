# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .custom_single_stage import customSingleStageDetector


@MODELS.register_module()
class YOLOF(customSingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOF. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOF. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional):
            Model preprocessing config for processing the input data.
            it usually includes ``to_rgb``, ``pad_size_divisor``,
            ``pad_value``, ``mean`` and ``std``. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

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
