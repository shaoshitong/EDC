# wandb disabled
wandb enabled
wandb offline

CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 100 \
    --model "resnet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir ../recover/syn_data/CSDC_b5_ImageNet_1k_Recover_IPC_10 \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16_CSDC_b5

CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_rn50_fkd' \
    --batch-size 100 \
    --model "resnet50" \
    --cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn50_fkd/ \
    --train-dir ../recover/syn_data/CSDC_b5_ImageNet_1k_Recover_IPC_10 \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16_CSDC_b5

CUDA_VISIBLE_DEVICES=0 python train_FKD_parallel.py \
    --wandb-project 'final_rn101_fkd' \
    --batch-size 100 \
    --model "resnet101" \
    --cos --loss-type "mse_gt" --ce-weight 0.025 \
    -j 4 --gradient-accumulation-steps 1  --st 2 \
    -T 20 --gpu-id 0 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn101_fkd/ \
    --train-dir ../recover/syn_data/CSDC_b5_ImageNet_1k_Recover_IPC_10 \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16_CSDC_b5