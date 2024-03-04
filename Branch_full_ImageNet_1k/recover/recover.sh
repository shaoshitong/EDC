CUDA_VISIBLE_DEVICES=0,1,2,3 python recover.py \
    --arch-name "resnet18" \
    --exp-name "CSDC_b5_closeness_IPC_40" \
    --batch-size 80 \
    --lr 0.05 --category-aware "local" \
    --ipc-number 40 --training-momentum 0.8  \
    --iteration 1000 --drop-rate 0.4 \
    --train-data-path /home/sst/imagenet/train/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1,2,3 --initial-img-dir /home/sst/product/RDED/exp/imagenet-1k_resnet18_f2_mipc300_ipc50_cr5/syn_data/