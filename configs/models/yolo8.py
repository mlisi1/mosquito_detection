_base_ = ['/mmdetection/configs/runtimes/yolov8_runtime.py', 
          '/mmdetection/configs/schedules/500e_yolov8_schedule.py', 
          '/mmdetection/configs/datasets/yolo_unified_mosquito_dataset.py'
          ]


# model = dict(
#     backbone=dict(
#         act_cfg=dict(inplace=True, type='SiLU'),
#         arch='P5',
#         deepen_factor=0.33,
#         last_stage_out_channels=1024,
#         norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#         type='YOLOv8CSPDarknet',
#         widen_factor=0.5,
#         init_cfg = dict(type='Pretrained', checkpoint='/workspace/trainings/yolov8_s.pth')
#         ),
#     bbox_head=dict(
#         bbox_coder=dict(type='DistancePointBBoxCoder'),
#         head_module=dict(
#             act_cfg=dict(inplace=True, type='SiLU'),
#             featmap_strides=[
#                 8,
#                 16,
#                 32,
#             ],
#             in_channels=[
#                 256,
#                 512,
#                 1024,
#             ],
#             norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#             num_classes=3,
#             reg_max=16,
#             type='YOLOv8HeadModule',
#             widen_factor=0.5),
#         loss_bbox=dict(
#             bbox_format='xyxy',
#             iou_mode='ciou',
#             loss_weight=7.5,
#             reduction='sum',
#             return_iou=False,
#             type='IoULoss'),
#         loss_cls=dict(
#             loss_weight=0.5,
#             reduction='none',
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True),
#         loss_dfl=dict(
#             loss_weight=0.375,
#             reduction='mean',
#             type='mmdet.DistributionFocalLoss'),
#         prior_generator=dict(
#             offset=0.5, strides=[
#                 8,
#                 16,
#                 32,
#             ], type='mmdet.MlvlPointGenerator'),
#         train_cfg=dict(
#             assigner=dict(num_classes=3, type='BatchTaskAlignedAssigner')),
#         type='YOLOv8Head'),
#     data_preprocessor=dict(
#         bgr_to_rgb=True,
#         mean=[
#             0.0,
#             0.0,
#             0.0,
#         ],
#         std=[
#             255.0,
#             255.0,
#             255.0,
#         ],
#         type='YOLOv5DetDataPreprocessor'),
#     neck=dict(
#         act_cfg=dict(inplace=True, type='SiLU'),
#         deepen_factor=0.33,
#         in_channels=[
#             256,
#             512,
#             1024,
#         ],
#         norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
#         num_csp_blocks=3,
#         out_channels=[
#             256,
#             512,
#             1024,
#         ],
#         type='YOLOv8PAFPN',
#         widen_factor=0.5),
#     test_cfg=dict(
#         max_per_img=300,
#         multi_label=True,
#         nms=dict(iou_threshold=0.7, type='nms'),
#         nms_pre=30000,
#         score_thr=0.001),
#     train_cfg=dict(
#         assigner=dict(
#             alpha=0.5,
#             beta=6.0,
#             eps=1e-09,
#             num_classes=3,
#             topk=10,
#             type='BatchTaskAlignedAssigner',
#             use_ciou=True)),
#     type='YOLODetector')




num_classes = 3

deepen_factor = 0.33
widen_factor = 0.5
strides = [8, 16, 32]
last_stage_out_channels = 1024
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
loss_dfl_weight = 1.5 / 4
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0 



model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300) 

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg = dict(type='Pretrained', checkpoint='/workspace/trainings/yolov8_s.pth')
        ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)