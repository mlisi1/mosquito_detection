auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
batch_size = 8
classes = (
    'Aedes',
    'Anopheles',
    'Culex',
)
data_root = 'data/unified_mosquito_dataset/'
dataset_type = 'CocoDataset'
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
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir='/workspace/trainings/retinanet_512_b8//test_imgs/',
        type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/workspace/trainings/retinanet_512_b8//best_coco_bbox_mAP_epoch_100.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=3,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=500,
        gamma=0.1,
        milestones=[
            80,
            140,
            180,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/'),
        data_root='data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/unified_mosquito_dataset/test/annotations.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(max_epochs=500, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        data_root='data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
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
                type='Rotate'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Brightness'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Contrast'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Sharpness'),
            dict(max_mag=1.2, min_mag=0.2, prob=0.6, type='Color'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
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
        type='Rotate'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Brightness'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Contrast'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Sharpness'),
    dict(max_mag=1.2, min_mag=0.2, prob=0.6, type='Color'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        data_root='data/unified_mosquito_dataset/',
        metainfo=dict(classes=(
            'Aedes',
            'Anopheles',
            'Culex',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/unified_mosquito_dataset/val/annotations.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
visualizer = dict(type='mmdet.DetLocalVisualizer')
work_dir = '/workspace/trainings/retinanet_512_b8'
