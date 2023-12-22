model = dict(
    type='AGNET',
    backbone=dict(
        type='g_resnet18',
        pretrained=True
    ),
    neck=dict(
        type='WFEN',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PA_Head',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)
data = dict(
    batch_size=12,
    train=dict(
        type='AGNET_Synth',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.5,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=1,
    optimizer='Adam'
)
