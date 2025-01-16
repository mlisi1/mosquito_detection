dataset_type = 'CocoDataset'
classes = ('Aedes', 'Anopheles', 'Culex',)
data_root='data/unified_mosquito_dataset/'

visualizer = dict(
    type='DetLocalVisualizer',

    name='visualizer')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Rotate', min_mag=0.0, max_mag=180.0, interpolation="bicubic", img_border_value=(0,0,0), prob=0.7),
    dict(type='Brightness', min_mag=0.2, max_mag=1.9, prob=0.4),     # Random brightness adjustment
    dict(type='Contrast', min_mag=0.2, max_mag=1.9, prob=0.4),       # Random contrast adjustment
    dict(type='Sharpness', min_mag=0.2, max_mag=1.9, prob=0.4),
    dict(type='Color', min_mag=0.2, max_mag=1.9, prob=0.6),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Rotate', min_mag=0.0, max_mag=180.0, interpolation="bicubic", img_border_value=(0,0,0), prob=0.5),
    dict(type='Brightness', min_mag=0.2, max_mag=1.9, prob=0.7),     # Random brightness adjustment
    dict(type='Contrast', min_mag=0.2, max_mag=1.9, prob=0.7),       # Random contrast adjustment
    dict(type='Sharpness', min_mag=0.2, max_mag=1.9, prob=0.7),
    dict(type='Color', min_mag=0.2, max_mag=1.9, prob=0.7),
    dict(type='PackDetInputs')
]

test_pipeline = val_pipeline


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        pipeline = train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)


val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        pipeline = test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)


test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/'),
        pipeline = test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)



val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotations.json',
    metric='bbox',
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/annotations.json',
    metric='bbox',
    format_only=False,
)