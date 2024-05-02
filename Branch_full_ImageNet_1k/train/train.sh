# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_efficientnet_b0_fkd' \
    --batch-size 100 \
    --model "efficientnet_b0" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_efficientnet_b0_fkd/ \
    --train-dir '../recover/syn_data/EDC_ImageNet_1k_Recover_IPC_10/' \
    --val-dir /data/imagenet1k/val/ \
    --fkd-path ../relabel/FKD_EDC_IPC_10