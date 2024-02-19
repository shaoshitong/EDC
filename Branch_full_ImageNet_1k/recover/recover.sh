CUDA_VISIBLE_DEVICES=1,4,5,6 python closeness.py \
    --arch-name "resnet18" \
    --exp-name "CSDC_b5_closeness_IPC_10" \
    --batch-size 80 \
    --lr 0.05 \
    --ipc-number 10 --training-momentum 0.8 --closeness \
    --iteration 2000 \
    --train-data-path /home/LargeData/Large/ImageNet/train/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 1,4,5,6 --initial-img-dir /home/chenhuanran2022/work/RDED/exp/imagenet-1k_resnet18_f2_mipc300_ipc10_cr5/syn_data/