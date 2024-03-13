CUDA_VISIBLE_DEVICES=0,1,2,3 python recover.py \
    --arch-name "resnet18" \
    --exp-name "CSDC_2nd_distill_IPC_10" \
    --batch-size 80 \
    --lr 0.05 --category-aware "local" \
    --ipc-number 10 --training-momentum 0.8  \
    --iteration 1000 --drop-rate 0.0 \
    --train-data-path /home/sst/product/CSDC/Branch_full_ImageNet_1k/recover/syn_data/CSDC_b5_closeness_IPC_50 \
    --l2-scale 0 --tv-l2 0 --r-loss 0.1 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1,2,3 --initial-img-dir /home/sst/product/RDED/exp/imagenet-1k_resnet18_f2_mipc300_ipc50_cr5/syn_data/