dataset_type = 'YOLOv5CocoDataset'
classes = ('Aedes', 'Anopheles', 'Culex',)
data_root='/mmdetection/data/unified_mosquito_dataset/'
backend_args = None
batch_size = 16

# visualizer = dict(
#     type='mmdet.DetLocalVisualizer',

#     name='visualizer')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmdet.Rotate', min_mag=0.0, max_mag=180.0, interpolation="bicubic", img_border_value=(0,0,0), prob=0.7),
    dict(type='mmdet.Brightness', min_mag=0.2, max_mag=1.2, prob=0.4),     # Random brightness adjustment
    dict(type='mmdet.Contrast', min_mag=0.2, max_mag=1.2, prob=0.4),       # Random contrast adjustment
    dict(type='mmdet.Sharpness', min_mag=0.2, max_mag=1.2, prob=0.4),
    dict(type='mmdet.Color', min_mag=0.2, max_mag=1.2, prob=0.4),
    dict(type='mmdet.PackDetInputs'),
    
]




val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='YOLOv5KeepRatioResize', scale=(512, 512), keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=False,
    drop_last = False,
    collate_fn=dict(type='yolov5_collate'),
    pin_memory = False,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        pipeline = train_pipeline,
    ),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    sampler=dict(type='DefaultSampler', shuffle=True)
)


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=False,
    drop_last = False,
    pin_memory = False,
    # collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        pipeline = test_pipeline,
        
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)


test_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    drop_last = False,
    persistent_workers=False,
    # collate_fn=dict(type='yolov5_collate'),
    pin_memory = False,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/'),
        pipeline = test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)



val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'val/annotations.json',
    metric='bbox',
    format_only=False,
)
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test/annotations.json',
    metric='bbox',
    format_only=False,
)