CUDA_VISIBLE_DEVICES=0 python data_synthesis_without_optim.py \
    --exp-name "WO_OPTIM_ImageNet_1k_Recover_IPC_10" \
    --ipc-number 10 \
    --train-data-path /path/to/imagenet-1k/train --gpu-id 0

CUDA_VISIBLE_DEVICES=0 python data_synthesis_with_svd_with_db_with_all_statistic.py \
    --arch-name "resnet18" \
    --exp-name "GVBSM_ImageNet_1k_Recover_IPC_10" \
    --batch-size 80 \
    --lr 0.2 \
    --ipc-number 10 \
    --iteration 4000 \
    --train-data-path /path/to/imagenet-1k/train \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0 --initial-img-dir ./syn_data/WO_OPTIM_ImageNet_1k_Recover_IPC_10
