CUDA_VISIBLE_DEVICES=0 python recover.py \
    --arch-name "resnet18" \
    --exp-name "EDC_CIFAR_10_Recover_IPC_1_backbone_8" \
    --batch-size 10 --category-aware "global" \
    --lr 0.05 --drop-rate 0.0 \
    --ipc-number 1 --training-momentum 0.8 \
    --iteration 75 \
    --train-data-path /data/cifar10/train/ \
    --r-loss 0.01 --initial-img-dir /data/cifar10/train/ \
    --verifier --store-best-images --gpu-id 0