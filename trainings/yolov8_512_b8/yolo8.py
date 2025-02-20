backend_args = None
batch_size = 8
classes = (
    'Aedes',
    'Anopheles',
    'Culex',
)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_root = '/mmdetection/data/unified_mosquito_dataset/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.33
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=5,
        max_keep_ckpts=3,
        rule='greater',
        save_best='auto',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(
        interval=100, out_suffix=[
            '.log',
            '.json',
        ], type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=500,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir='/workspace/trainings/yolov8_512_b8//test_imgs/',
        type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
last_stage_out_channels = 1024
launcher = 'none'
load_from = '/workspace/trainings/yolov8_512_b8//best_coco_bbox_mAP_epoch_184.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=0.33,
        init_cfg=dict(
            checkpoint='/workspace/trainings/yolov8_s.pth', type='Pretrained'),
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=3,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8PAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=3,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=16,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
resume = False
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/'),
        data_root='/mmdetection/data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                scale=(
                    512,
                    512,
                ),
                type='YOLOv5KeepRatioResize'),
            dict(type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/mmdetection/data/unified_mosquito_dataset/test/annotations.json',
    format_only=False,
    metric='bbox',
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='YOLOv5KeepRatioResize'),
    dict(type='mmdet.PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            490,
            1,
        ),
    ],
    max_epochs=500,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    batch_size=8,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        data_root='/mmdetection/data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                scale=(
                    512,
                    512,
                ),
                type='YOLOv5KeepRatioResize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                img_border_value=(
                    0,
                    0,
                    0,
                ),
                interpolation='bicubic',
                max_mag=180.0,
                min_mag=0.0,
                prob=0.7,
                type='mmdet.Rotate'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Brightness'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Contrast'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Sharpness'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Color'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='YOLOv5KeepRatioResize'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        img_border_value=(
            0,
            0,
            0,
        ),
        interpolation='bicubic',
        max_mag=180.0,
        min_mag=0.0,
        prob=0.7,
        type='mmdet.Rotate'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Brightness'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Contrast'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Sharpness'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='mmdet.Color'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        data_root='/mmdetection/data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                scale=(
                    512,
                    512,
                ),
                type='YOLOv5KeepRatioResize'),
            dict(type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/mmdetection/data/unified_mosquito_dataset/val/annotations.json',
    format_only=False,
    metric='bbox',
    type='mmdet.CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='YOLOv5KeepRatioResize'),
    dict(type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(type='mmdet.DetLocalVisualizer')
widen_factor = 0.5
work_dir = '/workspace/trainings/yolov8_512_b8'
