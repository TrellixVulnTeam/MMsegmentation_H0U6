_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/LeafDataset_128.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]
model = dict(
    decode_head=dict(num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[1,20])),
    auxiliary_head=dict(num_classes=2, align_corners=False,
                    ),
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))
# optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005)


