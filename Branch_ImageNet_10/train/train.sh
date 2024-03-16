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
    --train-dir '../recover/syn_data/ImageNet_10_IPC_50/' \
    --val-dir /home/sst/imagenet/val/ &
CUDA_VISIBLE_DEVICES=1 python direct_train.py \
    --wandb-project 'final_rn50_fkd' \
    --batch-size 50 \
    --model "resnet50" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 1 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn50_fkd/ \
    --train-dir '../recover/syn_data/ImageNet_10_IPC_50/' \
    --val-dir /home/sst/imagenet/val/ &
CUDA_VISIBLE_DEVICES=2 python direct_train.py \
    --wandb-project 'final_rn101_fkd' \
    --batch-size 50 \
    --model "resnet101" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 2 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn101_fkd/ \
    --train-dir '../recover/syn_data/ImageNet_10_IPC_50/' \
    --val-dir /home/sst/imagenet/val/ &
CUDA_VISIBLE_DEVICES=3 python direct_train.py \
    --wandb-project 'final_mobilenet_v2_fkd' \
    --batch-size 50 \
    --model "mobilenet_v2" \
    --ls-type cos2 --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 3 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_mobilenet_v2_fkd/ \
    --train-dir '../recover/syn_data/ImageNet_10_IPC_50/' \
    --val-dir /home/sst/imagenet/val/