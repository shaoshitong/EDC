'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import random
import argparse
import collections
import time

from tqdm import tqdm
import numpy as np
import torchvision.datasets
from PIL import Image

import torch.multiprocessing as mp
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import multiprocessing
import torch.distributed as dist
mp.set_sharing_strategy('file_system')
from utils import *

def shift_list(lst, shift_count):
    new_lst = [i for i in range(shift_count,len(lst),1)]
    for i in range(shift_count):
        new_lst.append(i)
    return new_lst

def main_worker(gpu, ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    model_teacher = [_model_teacher.cuda(gpu).eval() for _model_teacher in model_teacher]

    for _model_teacher in model_teacher:
        for p in _model_teacher.parameters():
            p.requires_grad = False
    model_verifier = model_verifier.cuda(gpu)
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False
    hook_for_display = lambda x, y: validate(x, y, model_verifier)

    save_every = 20
    batch_size = args.batch_size
    best_cost = 1e4
    load_tag_dict = [False for i in range(len(model_teacher))]
    loss_r_feature_layers = [[] for _ in range(len(model_teacher))]
    load_tag = True

    for i, (_model_teacher) in enumerate(model_teacher):
        for name, module in _model_teacher.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers[i].append(BNFeatureHook(module,
                                                              training_momentum=args.training_momentum,
                                                              flatness_weight=args.flatness_weight))
            elif isinstance(module, nn.Conv2d):
                if args.aux_teacher[i] in ["wide_resnet50_2", "regnet_y_400mf", "regnet_x_400mf"]:
                    full_name = str(_model_teacher.__class__.__name__) + "_" + str(args.aux_teacher[i]) + "=" + name
                else:
                    full_name = str(_model_teacher.__class__.__name__) + "=" + name
                _hook_module = ConvFeatureHook(module, save_path=args.statistic_path,
                                               name=full_name,
                                               gpu=gpu, training_momentum=args.training_momentum,
                                               drop_rate=args.drop_rate,
                                               flatness_weight=args.flatness_weight)
                _hook_module.set_hook(pre=True)
                load_tag = load_tag & _hook_module.load_tag
                load_tag_dict[i] = _hook_module.load_tag
                loss_r_feature_layers[i].append(_hook_module)

    sub_batch_size = int(batch_size // ngpus_per_node)

    initial_img_cache = PreImgPathCache(args.initial_img_dir,transforms=transforms.Compose([
                                                             transforms.Resize((224,224)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])]))
    if not load_tag:
        train_dataset = torchvision.datasets.ImageFolder(root=args.train_data_path,
                                                         transform=transforms.Compose([
                                                             transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])]))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   num_workers=4,
                                                   batch_size=256,
                                                   drop_last=False,
                                                   shuffle=True)

        with torch.no_grad():
            for j, _model_teacher in enumerate(model_teacher):
                if not load_tag_dict[j]:
                    print(f"conduct backbone {args.aux_teacher[j]} statistics")
                    for i, (data, _) in tqdm(enumerate(train_loader)):
                        data = data.cuda(gpu)
                        _ = _model_teacher(data)
                    for _loss_t_feature_layer in loss_r_feature_layers[j]:
                        if isinstance(_loss_t_feature_layer, ConvFeatureHook):
                            _loss_t_feature_layer.save()

        print("Training Statistic Information Is Successfully Saved")
    else:
        print("Training Statistic Information Is Successfully Load")

    for j in range(len(loss_r_feature_layers)):
        for _loss_t_feature_layer in loss_r_feature_layers[j]:
            if isinstance(_loss_t_feature_layer, ConvFeatureHook):
                _loss_t_feature_layer.set_hook(pre=False)

    targets_all_all = torch.LongTensor(np.arange(1000))[None, ...].expand(len(ipc_id_range), 1000).contiguous().view(-1)
    ipc_id_all = torch.LongTensor(ipc_id_range)[..., None].expand(len(ipc_id_range), 1000).contiguous().view(-1)

    total_number = 1000 * (ipc_id_range[-1] + 1 - ipc_id_range[0])
    turn_index = torch.LongTensor(np.arange(total_number)).view(len(ipc_id_range), 1000) \
        .transpose(1, 0).contiguous().view(-1)

    counter = 0
    for zz in range(0, total_number, batch_size):  # 9900 - 10000
        sub_turn_index = turn_index[zz + gpu * sub_batch_size:min(zz + (gpu + 1) * sub_batch_size, total_number)]
        targets = targets_all_all[sub_turn_index].cuda(gpu)
        ipc_ids = ipc_id_all[sub_turn_index].cuda(gpu)

        data_type = torch.float
        sub_batch_size = min(zz + (gpu + 1) * sub_batch_size, total_number) - (zz + gpu * sub_batch_size)
        if sub_batch_size < 0:
            continue
        print(f"In GPU {gpu}, targets is set as: \n{targets}\n, ipc_ids is set as: \n{ipc_ids}")

        if args.initial_img_dir is not None:
            inputs = torch.stack([initial_img_cache.random_img_sample(_target) for _target in targets.tolist()],0).to(f'cuda:{gpu}').to(data_type)
            inputs.requires_grad_(True)
        else:
            inputs = torch.randn((sub_batch_size, 3, 224, 224), requires_grad=True, device=f'cuda:{gpu}',
                                 dtype=data_type)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter, args.jitter
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        inputs_ema = EMA(alpha=args.ema_alpha, initial_value=inputs)
        backbone_ema_dict = [EMA(alpha=args.ema_alpha, initial_value=inputs) for model_name in args.aux_teacher]
        backbone_ema_dict_tmp = [EMA(alpha=args.ema_alpha, initial_value=inputs) for model_name in args.aux_teacher]
        cosine_similarity_cache = []

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)
            inputs_ema_jit = aug_function(inputs_ema.value)

            # apply random jitter offsets
            # off1 = random.randint(0, lim_0)
            # off2 = random.randint(0, lim_1)
            # inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
            # inputs_ema_jit = torch.roll(inputs_ema_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            id = counter % len(model_teacher)
            counter += 1
            if args.flatness:
                with torch.no_grad():
                    for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                        mod.set_ema()
                    ema_sub_outputs = model_teacher[id](inputs_ema_jit)
            for (idx, mod) in enumerate(loss_r_feature_layers[id]):
                mod.set_ori(flatness=args.flatness)
            sub_outputs = model_teacher[id](inputs_jit)

            with torch.no_grad():
                true_dist = torch.zeros_like(sub_outputs)
                true_dist.fill_(0.1 / (1000 - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 0.9)
                true_dist = true_dist + (torch.rand_like(true_dist) * 0.1 / (1000 - 1) - 0.05 / (1000 - 1))
                true_dist = true_dist.abs()

            # R_cross classification loss
            loss_ce = criterion(sub_outputs, targets)

            # R_feature loss
            rescale = [args.first_multiplier] + [1. for _ in range(len(loss_r_feature_layers[id]) - 1)]
            loss_r_feature = sum(
                [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers[id])])

            if args.flatness:
                loss_ema_ce = F.kl_div(torch.log_softmax(sub_outputs / 4, dim=1),
                                        torch.softmax(ema_sub_outputs / 4, dim=1))
            else:
                loss_ema_ce = torch.Tensor([0.]).to(inputs_jit.device)
            if args.closeness and (iteration >= (args.iteration // 20)): # (args.iteration // 10)
                with torch.no_grad():
                    idxs = shift_list(backbone_ema_dict,id)
                    begins = range(len(backbone_ema_dict)-1,0,-1)
                    differential = []
                    
                    for begin in begins:
                        differential.append(inputs - args.ema_alpha*(backbone_ema_dict[idxs[begin]].value - backbone_ema_dict[idxs[begin-1]].value)/(1-args.ema_alpha))
                    differential.append(inputs - args.ema_alpha*(backbone_ema_dict[idxs[0]].value-backbone_ema_dict_tmp[idxs[-1]].value)/(1-args.ema_alpha))
                
                loss_closeness = 0
                for _differential in differential:
                    loss_closeness +=  (1/len(backbone_ema_dict)) * F.mse_loss(inputs, _differential)
            else:
                loss_closeness = torch.Tensor([0.]).to(inputs_jit.device)

            # Nuclear losses
            adaptivepool = nn.AdaptiveAvgPool2d((32, 32))
            re_inputs_jit = adaptivepool(inputs_jit).reshape(inputs_jit.shape[0], -1)
            if zz + ngpus_per_node * sub_batch_size > total_number:
                dist_inputs_jit = re_inputs_jit
            else:
                dist_inputs_jit = GatherLayer.apply(re_inputs_jit)
                dist_inputs_jit = torch.cat(dist_inputs_jit, 0)

            aux_nuc_norm_loss = 0.
            _targets = targets_all_all[
                turn_index[zz: min(zz + ngpus_per_node * sub_batch_size, total_number)]].cuda(gpu).int()
            set_targets = set(_targets.tolist())
            for i in set_targets:
                sub_class_inputs_jit = dist_inputs_jit[_targets == i]
                sub_class_inputs_jit = sub_class_inputs_jit @ sub_class_inputs_jit.t()
                l = torch.linalg.eigvals(sub_class_inputs_jit).real.float()
                stu = l.log_softmax(dim=-1)
                tea = (l / args.tau).softmax(dim=-1)
                aux_nuc_norm_loss += nn.KLDivLoss(reduction="sum")(stu, tea.detach())
            nuc_norm = aux_nuc_norm_loss

            # nuc_norm = tr(S), where U^TSV = input_jit

            # combining losses
            loss_aux = args.nuc_norm * nuc_norm + \
                       args.r_loss * loss_r_feature

            loss = loss_ce + loss_aux + loss_ema_ce * args.flatness_weight + loss_closeness * args.closeness_weight

            if iteration % save_every == 0 and args.gpu == 0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("nuc norm loss", (args.nuc_norm * nuc_norm).item())
                print("loss_r_feature", loss_r_feature.item())
                print("loss_ema_ce", loss_ema_ce.item())
                print("loss_closeness", loss_closeness.item())
                print("main criterion",
                      criterion(sub_outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            if args.average_grad_ratio > 0:
                grad = inputs_jit.grad.mean(0)
                grad = dist.all_reduce(grad, async_op=True) / ngpus_per_node
                inputs_jit.grad = (1 - args.average_grad_ratio) * inputs_jit.grad + (
                    args.average_grad_ratio) * grad
                
            with torch.no_grad():
                _grad = inputs.grad.view(-1)
                cosine_similarity_cache.append(_grad/torch.sqrt((_grad**2).sum()))
                if len(cosine_similarity_cache)>len(args.aux_teacher):
                    cosine_similarity_cache.pop(0)
                    cosine_similarity_value = torch.stack(cosine_similarity_cache,0) # N,K 
                    cosine_similarity_value = cosine_similarity_value @ cosine_similarity_value.T
                    cosine_similarity_value = cosine_similarity_value[~torch.eye(cosine_similarity_value.shape[0],device=cosine_similarity_value.device).bool()].sum().item()/(
                        cosine_similarity_value.shape[0]*cosine_similarity_value.shape[0]-cosine_similarity_value.shape[0])
                    if iteration % save_every == 0 and args.gpu == 0:
                        print("cosine_similarity", cosine_similarity_value)
                # inputs.grad = inputs.grad/inputs.grad.norm()
            optimizer.step()

            with torch.no_grad(): # 0^ 1^ 2 3 0^ 1^ 2^ 3^
                inputs_ema.ema_update(inputs)
                backbone_ema_dict[id].ema_update(inputs)
                backbone_ema_dict_tmp[(id+len(args.aux_teacher)-2)%len(args.aux_teacher)].value = \
                    backbone_ema_dict[(id+len(args.aux_teacher)-2)%len(args.aux_teacher)].value
            
            # clip color outlayers
            inputs.data = clip(inputs.data)
            if gpu == 0 and (best_cost > loss.item() or iteration == 1):
                best_inputs = inputs.data.clone()

        del inputs_ema
        del backbone_ema_dict
        del backbone_ema_dict_tmp

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_ids)
        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
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


def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def main_syn():
    parser = argparse.ArgumentParser(
        "G-VBSM: applying generalized matching for data condensation")
    """Data save flags"""
    parser.add_argument('--flatness', action='store_true', default=False,
                        help='encourage the flatness or not')
    parser.add_argument('--flatness-weight', type=float, default=0.25,
                        help='the weight of flatness weight')
    parser.add_argument('--closeness', action='store_true', default=False,
                        help='encourage the closeness or not')
    parser.add_argument('--closeness-weight', type=float, default=25,
                        help='the weight of closeness weight')
    parser.add_argument('--ema_alpha', type=float, default=0.9,
                        help='the weight of EMA learning rate')
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--ipc-number', type=int, default=50, help='the number of each ipc')
    parser.add_argument('--initial-img-dir', type=str, default="./syn_data/WO_OPTIM_ImageNet_1k_Recover_IPC_10", help="imgs used for initialization")
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--gpu-id', type=str, default='0,1')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-loss', type=float, default=0.05,
                        help='coefficient for BN and Conv feature distribution regularization')
    parser.add_argument('--first-multiplier', type=float, default=10.,
                        help='additional multiplier on first layer of L_bn or L_conv')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--training-momentum', type=float, default=0.4,
                        help="$\alpha$ in our paper, controls the form of score distillation sampling")
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help="$\beta_\textrm{dr}$ in our paper, controls the efficiency of GSM")
    parser.add_argument('--nuc-norm', type=float, default=0.00001,
                        help='coefficient for total variation Nuclear loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--tau', type=float, default=4.0, help='the temperature of nuc norm')
    parser.add_argument('--average_grad_ratio', default=0., type=float)
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--train-data-path', type=str, default='./imagenet/train',
                        help="the path of the ImageNet-1k's training set")
    parser.add_argument('--statistic-path', type=str, default='./statistic',
                        help="the path of the statistic file")
    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    aux_teacher = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5", "alexnet"] # "densenet121", "convnext_tiny"
    args.aux_teacher = aux_teacher
    model_teacher = []
    for name in aux_teacher:
        model_teacher.append(models.__dict__[name](pretrained=True))

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    ipc_id_range = list(range(0, args.ipc_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10000 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node,
             args=(ngpus_per_node, args, model_teacher, model_verifier, ipc_id_range))


if __name__ == '__main__':
    main_syn()
