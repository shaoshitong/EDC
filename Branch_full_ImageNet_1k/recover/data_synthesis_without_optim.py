'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import argparse
import os

import PIL.Image
from tqdm import tqdm
import numpy as np
import torchvision.datasets
from PIL import Image
import torch.multiprocessing as mp
import torch
import einops
import torch.utils
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data.distributed
import torch.distributed as dist
mp.set_sharing_strategy('file_system')

"""
CUDA_VISIBLE_DEVICES=0 python data_synthesis_without_optim.py \
    --exp-name "WO_OPTIM_ImageNet_1k_Recover_IPC_10" \
    --ipc-number 10 \
    --train-data-path /home/Bigdata/imagenet/train --gpu-id 0
"""


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def main_worker(gpu, ngpus_per_node, args, model_teacher):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    model_teacher = [_model_teacher.cuda(gpu).eval() for _model_teacher in model_teacher]

    train_dataset = torchvision.datasets.ImageFolder(root=args.train_data_path,
                                                     transform=transforms.Compose([
                                                         transforms.Resize((224, 224)),
                                                         # transforms.RandomHorizontalFlip(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=4,
                                               batch_size=256,
                                               drop_last=False,
                                               shuffle=False)

    patch_operator = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip()])
    loss_function = nn.CrossEntropyLoss(reduction="none")
    pre_patch_memory = [[] for _ in range(1000)]

    iter_number = 1281167 / 256
    print("Begin Pre-Selecting Images in the training dataset")
    if os.path.exists("./pre_patch_memory.npz"):
        pre_patch_memory = np.load("./pre_patch_memory.npz")["pre_patch_memory"]

    else:
        with torch.no_grad():
            for i, (data, label) in enumerate(train_loader):
                print(f"Pass {i * 100 / iter_number}%", end="\r")
                data = data.cuda(gpu)
                label = label.cuda(gpu)
                total_output = []
                for j, _model_teacher in enumerate(model_teacher):
                    output = _model_teacher(data)
                    total_output.append(output)
                total_output = torch.stack(total_output, 0).mean(0)
                loss = loss_function(total_output, label)
                for j in range(data.shape[0]):
                    _loss = loss[j].item()
                    _label = label[j].item()
                    pre_patch_memory[_label].append([i * 256 + j, _loss])

        for i in range(len(pre_patch_memory)):
            if len(pre_patch_memory[i]) != 0:
                pre_patch_memory[i] = [kk[0] for kk in sorted(pre_patch_memory[i], key=lambda x: x[1])[:300]]

        pre_patch_memory = np.array(pre_patch_memory)
        np.savez("./pre_patch_memory.npz", pre_patch_memory=pre_patch_memory)

    print("Begin Post-Selecting Images in the training dataset")

    intermediate_path = "./intermediate_path/"

    if not os.path.exists(intermediate_path):
        new_patch_memory = [[] for _ in range(1000)]
        with torch.no_grad():
            for i in tqdm(range(pre_patch_memory.shape[0])):
                counter = 0
                index_list = pre_patch_memory[i].tolist()
                total_data, total_label = [], []
                for j in index_list:
                    data, label = train_dataset[j]
                    total_data.append(data)
                    total_label.append(torch.LongTensor([i]))
                total_data = torch.stack(total_data, 0)
                total_label = torch.stack(total_label, 0)
                for j in range(int(total_data.shape[0] // 64)):
                    local_data = total_data[j * 64:j * 64 + 64].cuda()
                    local_label = total_label[j * 64:j * 64 + 64].cuda()
                    local_patch_data = []
                    local_patch_label = []
                    for iter in range(4):
                        patch = patch_operator(local_data)
                        local_patch_data.append(patch)
                        local_patch_label.append(local_label)
                    local_patch_data = torch.cat(local_patch_data, 0)
                    local_patch_label = torch.cat(local_patch_label, 0)
                    total_output = []
                    for j, _model_teacher in enumerate(model_teacher):
                        output = _model_teacher(local_patch_data)
                        total_output.append(output)
                    total_output = torch.stack(total_output, 0).mean(0)
                    local_patch_loss = loss_function(total_output, local_patch_label[..., 0])
                    local_patch_data = denormalize(local_patch_data)
                    for j in range(local_patch_data.shape[0]):
                        _loss = local_patch_loss[j].item()
                        _label = local_patch_label[j].item()
                        # save into separate folders
                        dir_path = '{}/new{:03d}'.format(intermediate_path, _label)
                        place_to_store = dir_path + '/loss{}_id{:03d}.jpg'.format(round(_loss, 12), counter)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path, exist_ok=True)
                        image_np = local_patch_data[j].data.cpu().numpy().transpose((1, 2, 0))
                        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
                        pil_image.save(place_to_store)

    print("Begin Image Synthetic from the candidate list")

    post_transform = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

    import glob
    for i in tqdm(range(1000)):
        dir_path = '{}/new{:03d}'.format(intermediate_path, i)
        image_paths = glob.glob(dir_path + "/*.jpg")
        new_image_paths = sorted(image_paths, key=lambda x: float(x.split("_id")[0].split("loss")[1]))
        new_image_paths = new_image_paths[:args.ipc_number * 4]
        total_image = []
        for image_path in new_image_paths:
            _image = PIL.Image.open(image_path).convert("RGB")
            _image = post_transform(_image)
            total_image.append(_image.cuda())
        total_image = torch.stack(total_image, 0)
        total_image = einops.rearrange(total_image, "(i n m) c h w -> i c (n h) (m w)", n=2, m=2)
        print(total_image.shape)
        labels = torch.ones(total_image.shape[0]).to(total_image.device) * i  # (IPC,)
        labels = labels.int()
        images = total_image  # (IPC,C,H,W)
        ipc_ids = [j for j in range(total_image.shape[0])]
        save_images(args, images, labels, ipc_ids)

    torch.cuda.empty_cache()


def save_images(args, images, targets, ipc_ids):
    ipc_id_range = ipc_ids
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_id{:03d}.jpg'.format(class_id, ipc_id_range[id])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main_syn():
    parser = argparse.ArgumentParser(
        "On the diversify and realism of distilled dataset")
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--ipc-number', type=int, default=50, help='the number of each ipc')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    """Optimization related flags"""
    parser.add_argument('--gpu-id', type=str, default='0,1')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    """Model related flags"""
    parser.add_argument('--train-data-path', type=str, default='./imagenet/train',
                        help="the path of the ImageNet-1k's training set")
    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    aux_teacher = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5"]  # "densenet121
    model_teacher = []
    for name in aux_teacher:
        model_teacher.append(models.__dict__[name](pretrained=True))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10001 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node,
             args=(ngpus_per_node, args, model_teacher))


if __name__ == '__main__':
    main_syn()
