auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
batch_size = 8
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'
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
        test_out_dir='/workspace/trainings/RTMDet_512_b8//test_imgs/',
        type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/workspace/trainings/RTMDet_512_b8//best_coco_bbox_mAP_epoch_357.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.5),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=False,
        feat_channels=128,
        in_channels=128,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=3,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            128,
            256,
            512,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=128,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0005),
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
            dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Color'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
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
    dict(max_mag=1.2, min_mag=0.2, prob=0.4, type='Color'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
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
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
visualizer = dict(type='mmdet.DetLocalVisualizer')
work_dir = '/workspace/trainings/RTMDet_512_b8'
