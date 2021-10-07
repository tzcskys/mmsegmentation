_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/fudan.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]
        )
)

# # AdamW optimizer, no weight decay for position embedding & layer norm
# # in backbone
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     # lr=0.00006,
#     lr=0.00001,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001
)

# scheduler = dict(
#     optimizer,
#     step_size=,
#     gamma=
# )

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu = 6,
    workers_per_gpu = 6
)

checkpoint_config = dict(
    by_epoch=False, interval=10000
)

evaluation = dict(
    interval= 500,
    metric='mDice')
