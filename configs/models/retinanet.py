_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../runtimes/save_best_runtime.py",
    "../datasets/unified_mosquito_dataset.py",
    "../schedules/500e_schedule.py"
]

# model settings
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        num_classes=3,
    ),
)
