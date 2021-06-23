_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='YOLOV5',
    backbone=dict(type='YOLOV5Backbone', depth_multiple=0.33, width_multiple=0.5),
    neck=None,
    bbox_head=dict(
        type='YOLOV5Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[0.33, 0.5, 1],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOV5BBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')
    ),
    test_cfg=dict(
        use_v3=False,  # 是否使用 mmdet v3 原本的后处理策略
        score_thr=0.05,  # 仅仅当use_v3 为 True 才有效
        nms_pre=1000,   # 仅仅当use_v3 为 True 才有效
        agnostic=False,  # 是否区分类别进行 nms，False 表示要区分
        multi_label=True,  # 是否考虑多标签， 单张图检测是为 False，test 时候为 True，可以提高 1 个点的 mAP
        min_bbox_size=0,
        # detect: conf_thr=0.25 iou_threshold=0.45
        # test: conf_thr=0.001 iou_threshold=0.65
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300)
)

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='ShapeLetterResize', img_scale=(640, 640), scaleup=True, auto=False),  # test
            # dict(type='LetterResize', img_scale=(640, 640), scaleup=True, auto=True),  # detect
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'pad_param'))
        ])
]

data = dict(
    test=dict(pipeline=test_pipeline))
