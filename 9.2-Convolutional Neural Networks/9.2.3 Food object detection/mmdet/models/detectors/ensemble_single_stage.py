# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector, customBaseDetector
import torch.nn as nn
import numpy as np
import torch
from mmcv.ops import nms
from mmdet.structures.det_data_sample import DetDataSample, InstanceData

@MODELS.register_module()
class ensembleSingleStageDetector(customBaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 backbone_d: ConfigType,
                 neck: OptConfigType = None,
                 neck_d: ConfigType = None,
                 neck_f: ConfigType = None,
                 bbox_head: ConfigType = None,
                 bbox_head_d: ConfigType = None,
                 bbox_head_f: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone_d = MODELS.build(backbone_d)

        if neck is not None:
            self.neck = MODELS.build(neck)
            self.neck_d = MODELS.build(neck_d)
            self.neck_f = MODELS.build(neck_f)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        bbox_head_d.update(train_cfg=train_cfg)
        bbox_head_d.update(test_cfg=test_cfg)

        bbox_head_f.update(train_cfg=train_cfg)
        bbox_head_f.update(test_cfg=test_cfg)

        self.bbox_head = MODELS.build(bbox_head)

        self.bbox_head_d = MODELS.build(bbox_head)

        self.bbox_head_f = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _init_layers(self):
        # self.fusion_conv = nn.Conv2d(4096, 2048, 1)
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(2048, 1024, 1),
            nn.Conv2d(4096, 2048, 1)
        ])
        # self.fusion_conv = nn.Conv2d(2560, 2048, 1)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_inputs_d: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # res = self.predict(batch_inputs, batch_inputs_d, batch_data_samples, True)
        # print("loss========================\n\n",batch_inputs.shape)
        # print(batch_inputs)
        # batch_data_samples += 1
        # x = self.extract_feat(batch_inputs, batch_inputs_d)
        # print(type(batch_inputs))
        # batch_inputs_d = np.array(batch_inputs_d.cpu())
        #
        # batch_inputs_d = torch.tensor(batch_inputs_d).float().cuda()
        # batch_inputs_d = torch.stack(batch_inputs_d).permute(0, 3, 1, 2)

        # print(len(batch_inputs_d))
        # print("\n\n\n\n\n\n==============")
        x = self.backbone(batch_inputs)
        x_d = self.backbone_d(batch_inputs_d)

        # print(self.backbone_d)
        # print(x[0].shape)
        # print("============x.shape",len(x))
        # print("============x_d.shape",x_d[0].shape)

        # num_channels = x[0].shape[1]

        x_f = []
        for i in range(len(x)):
            # print("============xi.shape", x[i].shape)
            concat = torch.cat((x[i], x_d[i]), dim=1)

            # print("============x_f.shape", concat.shape)
            # print(x[i].shape)
            concat = self.fusion_convs[i](concat)
            x_f.append(concat)
        # print("============x_f.shape",x_f[0].shape)
        x_f = tuple(x_f)

        if self.with_neck:
            x = self.neck(x)
            x_d = self.neck_d(x_d)
            x_f = self.neck_f(x_f)


        losses = self.bbox_head.loss(x, batch_data_samples)
        losses_d = self.bbox_head_d.loss(x_d, batch_data_samples)
        losses_f = self.bbox_head_f.loss(x_f, batch_data_samples)
        #
        losses['loss_cls'] = losses['loss_cls'] + losses_d['loss_cls'] + losses_f['loss_cls']
        losses['loss_bbox'] = losses['loss_bbox'] + losses_d['loss_bbox'] + losses_f['loss_bbox']

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_inputs_d: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # x = self.extract_feat(batch_inputs)
        # results_list = self.bbox_head.predict(
        #     x, batch_data_samples, rescale=rescale)
        # batch_data_samples = self.add_pred_to_datasample(
        #     batch_data_samples, results_list)
        # print(batch_inputs_d)
        # print("\n\n\n\n\n\n==============")
        # data = batch_inputs_d[0]
        # for t in batch_inputs_d[1:]:
        #     data = data.stack()
        # batch_inputs_d = np.array(batch_inputs_d.cpu())
        # batch_inputs_d = torch.stack(batch_inputs_d).permute(0, 3, 1, 2)


        # batch_inputs_d = torch.tensor(batch_inputs_d).float().cuda()

        x = self.backbone(batch_inputs)
        x_d = self.backbone_d(batch_inputs_d)

        x_f = []

        for i in range(len(x)):
            concat = torch.cat((x[i], x_d[i]), dim=1)
            concat = self.fusion_convs[i](concat)
            x_f.append(concat)
        x_f = tuple(x_f)

        if self.with_neck:
            x = self.neck(x)
            x_d = self.neck_d(x_d)
            x_f = self.neck_f(x_f)

        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        result = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        results_list_d = self.bbox_head_d.predict(
            x_d, batch_data_samples, rescale=rescale)
        result_d = self.add_pred_to_datasample(
            batch_data_samples, results_list_d)

        results_list_f = self.bbox_head_f.predict(
            x_f, batch_data_samples, rescale=rescale)
        result_f = self.add_pred_to_datasample(
            batch_data_samples, results_list_f)

        new_res = [DetDataSample(metainfo=result[i].metainfo) for i in range(len(result))]
        for i in range(len(result)):
            new_res[i].gt_instances = result[i].gt_instances
            new_res[i].ignored_instances = result[i].ignored_instances

            pred_instances = InstanceData(metainfo=result[i].metainfo)
            new_res[i].pred_instances = pred_instances

            # pred_instances.bboxes = torch.rand((5, 4))
            # pred_instances.scores = torch.rand((5,))
        weights = [1, 1, 0.2]
        for idx in range(len(result)):
            # print(result[idx].pred_instances.labels)
            # print(result['pred_instances'])
            # for i in range(len(result_d[idx].pred_instances.scores)):
            #     result[idx].pred_instances.scores.append(result_d[idx].pred_instances.scores[i])
            #     result[idx].pred_instances.labels.append(result_d[idx].pred_instances.labels[i])
            #     result[idx].pred_instances.bboxes.append(result_d[idx].pred_instances.bboxes[i])
            #
            # for i in range((result_f[idx].pred_instances.scores)):
            #     result[idx].pred_instances.scores.append(result_f[idx].pred_instances.scores[i])
            #     result[idx].pred_instances.labels.append(result_f[idx].pred_instances.labels[i])
            #     result[idx].pred_instances.bboxes.append(result_f[idx].pred_instances.bboxes[i])
            # ten = torch.cat((result[idx].pred_instances.scores,
            #                                                 result_d[idx].pred_instances.scores,
            #                                                 result_f[idx].pred_instances.scores), dim=0)

            all_scores = torch.cat((result[idx].pred_instances.scores * weights[0],
                                                            result_d[idx].pred_instances.scores * weights[1],
                                                            result_f[idx].pred_instances.scores * weights[2]), dim=0)

            all_boxes = torch.cat((result[idx].pred_instances.bboxes,
                                                            result_d[idx].pred_instances.bboxes,
                                                            result_f[idx].pred_instances.bboxes), dim=0)

            all_labels = torch.cat((result[idx].pred_instances.labels,
                                                            result_d[idx].pred_instances.labels,
                                                            result_f[idx].pred_instances.labels), dim=0)

            dets, keep = nms(all_boxes, all_scores, 0.5)

            final_boxes = all_boxes[keep]
            final_scores = all_scores[keep]
            final_labels = all_labels[keep]

            new_res[idx].pred_instances.scores = final_scores
            new_res[idx].pred_instances.bboxes = final_boxes
            new_res[idx].pred_instances.labels = final_labels
        # print(new_res)
        # print(results_list_f)
        # print(result_f)
        # print("=============================")
        # print(integrated_result)
        # print(result)
        #  = self.add_pred_to_datasample(
        #     batch_data_samples, new_res)
        return new_res

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_inputs_d: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor, batch_inputs_d: Tensor,) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        # print("extract_feat==========================", batch_inputs.shape)
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def integrate_detections(self, result, result_d, result_f, weights=[1.0, 1.0, 0.2], iou_threshold=0.5):
        all_boxes = []
        all_scores = []
        all_labels = []

        for res, weight in zip([result, result_d, result_f], weights):
            for sample in res:
                boxes = sample.pred_instances.bboxes
                scores = sample.pred_instances.scores * weight
                labels = sample.pred_instances.labels

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        dets, keep = nms(all_boxes, all_scores, iou_threshold)
        # print(type(keep))
        # print(keep)
        # keep = torch.tensor(keep, dtype=torch.long)  # 确保返回的是长整型张量

        final_boxes = all_boxes[keep]
        final_scores = all_scores[keep]
        final_labels = all_labels[keep]

        final_detections = {
            'pred_instances': {
                'bboxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            }
        }

        return final_detections

    # 假设 result, result_d, result_f 已经定义并且是有效的检测结果
