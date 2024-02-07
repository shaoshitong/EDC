# wandb disabled
wandb enabled
wandb offline

CUDA_VISBLE_DEVICES=0,1 python train_FKD_parallel.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 1024 \
    --model "resnet18" \
    --cos --loss-type "mse_gt" --ce-weight 0.1 \
    -j 4 --gradient-accumulation-steps 1 \
    -T 20 --gpu-id 0,1 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/ \
    --train-dir ../recover/syn_data/CSDC_b6_ImageNet_1k_Recover_IPC_10 \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16_CSDC_b6

CUDA_VISBLE_DEVICES=0,1,2,3 python train_FKD_parallel.py \
    --wandb-project 'final_rn50_fkd' \
    --batch-size 1024 \
    --model "resnet50" \
    --cos --loss-type "mse_gt" --ce-weight 0.1 \
    -j 4 --gradient-accumulation-steps 2 \
    -T 20 --gpu-id 0,1,2,3 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn50_fkd/ \
    --train-dir ../recover/syn_data/CSDC_b6_ImageNet_1k_Recover_IPC_10 \
    --val-dir /home/sst/imagenet/val/ \
    --fkd-path ../relabel/FKD_cutmix_fp16_CSDC_b6