# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=3 python direct_train.py \
    --wandb-project 'final_RN101_fkd' \
    --batch-size 100 --epochs 1000 \
    --model "ResNet101" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.0005 --gpu-id 3 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN101_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_Tiny_ImageNet_Recover_IPC_1 \
    --val-dir /home/sst/product/CSDC/Branch_Tiny_ImageNet/tiny-imagenet-200/