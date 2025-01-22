_base_ = [
    "../_base_/models/faster-rcnn_r50_fpn.py",
    "../runtimes/save_best_runtime.py",
    "../datasets/unified_mosquito_dataset.py",
    "../schedules/500e_schedule.py"
]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3,
        )
    )
)
