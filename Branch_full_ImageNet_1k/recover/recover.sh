CUDA_VISIBLE_DEVICES=0,1,2,3 python closeness.py \
    --arch-name "resnet18" \
    --exp-name "CSDC_b5_closeness2_IPC_10" \
    --batch-size 80 \
    --lr 0.05 \
    --ipc-number 10 --training-momentum 0.8 --closeness \
    --iteration 2000 \
    --train-data-path /home/sst/imagenet/train/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1,2,3 --initial-img-dir /home/sst/product/CSDC/Branch_full_ImageNet_1k/syn_data