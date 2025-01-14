

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Aedes aegypti', 'Culex quinquefasciatus',)
data_root='data/mosquito_detection_dataset/'


# vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    # vis_backends=vis_backends,
    name='visualizer')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackDetInputs')
]


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
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        pipeline = val_pipeline
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/'),
        pipeline = test_pipeline
        )
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
