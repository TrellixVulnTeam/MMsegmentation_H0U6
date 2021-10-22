_base_ = [
    'configs/_base_/models/deeplabv3plus_r50-d8.py',
    'configs/_base_/datasets/LeafDataset.py', 'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule_1k.py'
]
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[1,100])),
    auxiliary_head=dict(num_classes=2, align_corners=True,
                    ),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
# optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005)


