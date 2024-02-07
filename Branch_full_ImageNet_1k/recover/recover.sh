CUDA_VISIBLE_DEVICES=0,1,2,3 python data_synthesis_with_svd_with_db_with_all_statistic.py \
    --arch-name "resnet18" \
    --exp-name "CSDC_b6_ImageNet_1k_Recover_IPC_10" \
    --batch-size 80 \
    --lr 0.05 \
    --ipc-number 10 \
    --iteration 4000 \
    --train-data-path /home/sst/imagenet/train/ \
    --l2-scale 0 --tv-l2 0 --r-loss 0.01 --nuc-norm 1. \
    --verifier --store-best-images --gpu-id 0,1,2,3 --initial-img-dir ./syn_data/WO_OPTIM_ImageNet_1k_Recover_IPC_10
