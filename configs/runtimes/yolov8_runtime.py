default_scope = 'mmyolo'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, out_suffix=['.log', '.json']),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=500),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Frequency to save checkpoints (every epoch)
        by_epoch=True,  # Save checkpoints based on epoch (set False for iterations)
        save_best='auto',  # Automatically monitor the primary metric
        rule='greater',  # 'greater' if higher values are better, 'less' otherwise,
        max_keep_ckpts=3,  # Keep only the latest 3 checkpoints
        save_last=True,  # Always save the last checkpoint
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

launcher = 'none'
log_level = 'INFO'
load_from = None
resume = False

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions
# before MMDet 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

