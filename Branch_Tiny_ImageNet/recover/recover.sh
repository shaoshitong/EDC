
CUDA_VISIBLE_DEVICES=0,1,2,3 python recover.py \
    --arch-name "resnet18" \
    --exp-name "EDC_Tiny_ImageNet_Recover_IPC_10" \
    --batch-size 100 --category-aware "global" \
    --lr 0.05 --drop-rate 0.0 \
    --ipc-number 1 --training-momentum 0.8 \
    --iteration 2000 \
    --train-data-path /data/tinyimagenet/train/ \
    --r-loss 0.01 --initial-img-dir /data/tinyimagenet/train/ \
    --verifier --store-best-images --gpu-id 0,1,2,3