_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py',
    '../_base_/datasets/LeafDataset_128.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]

model = dict(
    decode_head=dict(num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[1,2])),
    auxiliary_head=dict(num_classes=2, align_corners=False,
                    ),
    test_cfg=dict(mode='slide', crop_size=(128, 128), stride=(112, 112)))
evaluation = dict(metric='mIoU')
