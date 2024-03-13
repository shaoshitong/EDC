# wandb disabled
wandb enabled
wandb offline


CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 100 \
    --model "resnet18" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir '../recover/syn_data/CSDC_2nd_distill_IPC_10/' \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_CSDC_IPC_10 &
CUDA_VISIBLE_DEVICES=1 python train_FKD_parallel.py \
    --wandb-project 'final_rn50_fkd' \
    --batch-size 100 \
    --model "resnet50" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 1 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn50_fkd/ \
    --train-dir '../recover/syn_data/CSDC_2nd_distill_IPC_10/' \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_CSDC_IPC_10 &
CUDA_VISIBLE_DEVICES=2 python train_FKD_parallel.py \
    --wandb-project 'final_rn101_fkd' \
    --batch-size 100 \
    --model "resnet101" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 2 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn101_fkd/ \
    --train-dir '../recover/syn_data/CSDC_2nd_distill_IPC_10/' \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_CSDC_IPC_10 &
CUDA_VISIBLE_DEVICES=3 python train_FKD_parallel.py \
    --wandb-project 'final_mobilenet_v2_fkd' \
    --batch-size 100 \
    --model "mobilenet_v2" \
    --ls-type cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 --ema-dr 0.99 \
    -T 20 --gpu-id 3 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_mobilenet_v2_fkd/ \
    --train-dir '../recover/syn_data/CSDC_2nd_distill_IPC_10/' \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_CSDC_IPC_10