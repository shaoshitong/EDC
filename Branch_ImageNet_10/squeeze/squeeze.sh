# "resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5", "alexnet"

CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --data_path /data/spiderman/imagenet/train/ &
CUDA_VISIBLE_DEVICES=1 python train.py --model mobilenet_v2 --data_path /data/spiderman/imagenet/train/ &
CUDA_VISIBLE_DEVICES=2 python train.py --model efficientnet_b0 --data_path /data/spiderman/imagenet/train/ &
CUDA_VISIBLE_DEVICES=3 python train.py --model shufflenet_v2_x0_5 --data_path /data/spiderman/imagenet/train/