CUDA_VISIBLE_DEVICES=0 python recover.py \
    --arch-name "resnet18" \
    --exp-name "EDC_ImageNet_10_Recover_IPC_10" \
    --batch-size 100 \
    --lr 0.05 --category-aware "global" \
    --ipc-number 50 --training-momentum 0.0  \
    --iteration 50 --drop-rate 0.0 \
    --train-data-path /data/imagenet10/train/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0 --initial-img-dir /data/imagenet10/RDED/ \