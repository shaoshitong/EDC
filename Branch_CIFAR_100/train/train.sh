# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=1 python direct_train.py \
    --wandb-project 'final_RN18_fkd' \
    --batch-size 50 --epochs 1000 \
    --model "ResNet18" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 1 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN18_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_100_Recover_IPC_50 \
    --val-dir '/home/LargeData/Regular/cifar/' &
CUDA_VISIBLE_DEVICES=4 python direct_train.py \
    --wandb-project 'final_RN50_fkd' \
    --batch-size 50 --epochs 1000 \
    --model "ResNet50" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 4 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN50_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_100_Recover_IPC_50 \
    --val-dir '/home/LargeData/Regular/cifar/' &
CUDA_VISIBLE_DEVICES=5 python direct_train.py \
    --wandb-project 'final_RN101_fkd' \
    --batch-size 50 --epochs 1000 \
    --model "ResNet101" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 5 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_RN101_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_100_Recover_IPC_50 \
    --val-dir '/home/LargeData/Regular/cifar/' &
CUDA_VISIBLE_DEVICES=6 python direct_train.py \
    --wandb-project 'final_MBV2_fkd' \
    --batch-size 50 --epochs 1000 \
    --model "MobileNetV2" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -T 20 --sgd --sgd-lr 0.1 --adamw-lr 0.001 --gpu-id 6 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    --mix-type 'cutmix' --weight-decay 0.0005 \
    --output-dir ./save/final_MBV2_fkd/ \
    --train-dir ../recover/syn_data/GVBSM_CIFAR_100_Recover_IPC_50 \
    --val-dir '/home/LargeData/Regular/cifar/'