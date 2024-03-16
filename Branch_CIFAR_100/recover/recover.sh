CUDA_VISIBLE_DEVICES=1,4,5,6 python recover.py \
    --arch-name "resnet18" \
    --exp-name "GVBSM_CIFAR_100_Recover_IPC_50" \
    --batch-size 100 --category-aware "global" \
    --lr 0.05 --drop-rate 0.0 \
    --ipc-number 50 --training-momentum 0.8 \
    --iteration 2000 \
    --train-data-path  /home/LargeData/Regular/cifar/ \
    --r-loss 0.01 --initial-img-dir /home/chenhuanran2022/work/RDED/exp/cifar100_resnet18_modified_f1_mipc300_ipc50_cr5/syn_data/ \
    --verifier --store-best-images --gpu-id 1,4,5,6