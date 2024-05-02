# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 10 --epochs 1000 \
    --model "ResNet18" \
    --ls-type multisteplr --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 0 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --adamw-weight-decay 0.01 \
    --output-dir ./save/final_RN18_fkd/ \
    --train-dir ../recover/syn_data/EDC_CIFAR_10_Recover_IPC_1_backbone_8/ \
    --val-dir '/data/cifar10/val/'