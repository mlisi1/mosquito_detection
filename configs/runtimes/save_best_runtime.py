default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, out_suffix=['.log', '.json']),
    param_scheduler=dict(type='ParamSchedulerHook'),
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
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
