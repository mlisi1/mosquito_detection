# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('Aedes', 'Anopheles', 'Culex',)
data_root='data/previous_work_dataset_converted/'

visualizer = dict(
    type='DetLocalVisualizer',

    name='visualizer')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(256, 256)),
    dict(type='PackDetInputs')
]

val_pipeline = train_pipeline
test_pipeline = train_pipeline


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
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/'),
        pipeline = val_pipeline,
    )
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
    )
)



