TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node 1 --master_port 2222 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root ~/DATA_ROOT/ \
    --val_freq 3 --batch_size 12 --save_freq 3 --print_freq 500 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset sr3d --test_dataset sr3d \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir ~/DATA_ROOT/output/logs/bdetr \
    --lr_decay_epochs 150 \
    --butd_cls --self_attend \
    --checkpoint_path ~/DATA_ROOT/checkpoints/SR3D_68_1.pth \
    --eval