# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_MobileNetV2_fkd' \
    --batch-size 100 \
    --model "MobileNetV2" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.0005 --gpu-id 0 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_MobileNetV2_fkd/ \
    --train-dir ../recover/syn_data/EDC_Tiny_ImageNet_Recover_IPC_10 \
    --val-dir /data/tinyimagenet/val/