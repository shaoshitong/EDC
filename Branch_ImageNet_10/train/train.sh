# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 50 \
    --model "resnet18" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir '../recover/syn_data/EDC_ImageNet_10_Recover_IPC_10/' \
    --val-dir /data/imagenet10/val/