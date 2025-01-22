dataset_type = 'CocoDataset'
classes = ('Aedes', 'Anopheles', 'Culex',)
data_root='data/unified_mosquito_dataset/'
backend_args = None
batch_size = 12

# visualizer = dict(
#     type='DetLocalVisualizer',

#     name='visualizer')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Rotate', min_mag=0.0, max_mag=180.0, interpolation="bicubic", img_border_value=(0,0,0), prob=0.7),
    dict(type='Brightness', min_mag=0.2, max_mag=1.2, prob=0.4),     # Random brightness adjustment
    dict(type='Contrast', min_mag=0.2, max_mag=1.2, prob=0.4),       # Random contrast adjustment
    dict(type='Sharpness', min_mag=0.2, max_mag=1.2, prob=0.4),
    dict(type='Color', min_mag=0.2, max_mag=1.2, prob=0.4),
    dict(type='PackDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='PackDetInputs')
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    drop_last = False,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/'),
        pipeline = train_pipeline,
    ),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    sampler=dict(type='DefaultSampler', shuffle=True)
)


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    drop_last = False,
    dataset=dict(
        type=dataset_type,
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
    dataset=dict(
        type=dataset_type,
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