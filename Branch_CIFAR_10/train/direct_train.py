import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb
import timm
from copy import deepcopy
import torch.distributed as dist
sys.path.append('../')
import relabel.models as ti_models
from baseline import get_network as ti_get_network
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
import torchvision.models as models
import torch.multiprocessing as mp
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy, get_parameters

normalize = transforms.Normalize([0.5071, 0.4867, 0.4408],
                                    [0.2675, 0.2565, 0.2761])

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

from relabel.utils_fkd import mix_aug

# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher

import torch
import torch.nn as nn


def cosine_similarity(a, b, eps=1e-5):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-5):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2, tem=4):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd

class EMAMODEL(object):
    def __init__(self,model):
        self.ema_model = deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self,decay_rate=0.999,model=None):
        for param,ema_param in zip(model.parameters(),self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)
    
    @torch.no_grad()
    def ema_swap(self,model=None):
        # print('Begin swap',list(self.ema_model.parameters())[0].data[0,0],list(model.parameters())[0].data[0,0])
        for param,ema_param in zip(self.ema_model.parameters(),model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp
        # print('After swap',list(self.ema_model.parameters())[0].data[0,0],list(model.parameters())[0].data[0,0])
    
    @torch.no_grad()
    def __call__(self, pre_z_t,t):
        return self.ema_model.module(pre_z_t,t)

class ALRS():
    def __init__(self, optimizer, decay_rate=0.95):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.prev_loss = 1e3

    def step(self, now_loss):
        delta = abs(self.prev_loss - now_loss)
        if delta / now_loss < 0.02 and delta < 0.02:
                self.optimizer.param_groups[0]["lr"] *= self.decay_rate
        self.p_lr = p_lr = self.optimizer.param_groups[0]["lr"]
        self.prev_loss = now_loss
        print(f"call auto learning rate scheduler, the learning rate is set as {p_lr}, the current loss is {now_loss}")

    def get_last_lr(self):
        return [self.p_lr]

def get_args():
    parser = argparse.ArgumentParser("FKD Training on ImageNet-1K")
    parser.add_argument('--batch-size', type=int,
                        default=100, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--epochs', type=int, default=1000, help='total epoch')
    parser.add_argument('--sgd-lr', type=float,
                        default=0.05, help='adamw learning rate')
    parser.add_argument('--pre-train-path', type=str,
                        default='../squeeze/squeeze_wo_ema/',
                        help='where to load the pre-trained backbone')
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--loss-type', type=str, default="kl",
                        help='the type of the loss function')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/imagenet/val', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')
    parser.add_argument('--ls-type', default="cos",
                        type=str, help='the type of lr scheduler')
    parser.add_argument('--alrs-dr', default=0.9975,
                        type=float, help='the decay rate of ALRS')
    parser.add_argument('--ema-dr', default=0.999,
                        type=float, help='the decay rate of EMA')
    parser.add_argument('--st', default=1.5,
                        type=float, help='the scheduler trick')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=1.024, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.875, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=3e-5, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.001, help='adamw learning rate')
    parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    parser.add_argument('--ce-weight', type=float,
                        default=0.1, help='the weight og cross-entropy loss')
    parser.add_argument('--gpu-id', type=str,
                        default="0,1", help='the id of gpu used')
    parser.add_argument('--model', type=str,
                        default='resnet18', help='student model name')
    parser.add_argument('--shuffle-patch', default=False, action='store_true',
                    help='if use shuffle-patch')
    parser.add_argument('--keep-topk', type=int, default=10,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    args.mode = 'fkd_load'
    return args


def main():
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    port_id = 10002 + np.random.randint(0, 1000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.set_start_method('spawn')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1])
    args.rank = gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    train_dataset = torchvision.datasets.ImageFolder(
        root=args.train_dir,
        transform=transforms.Compose([
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=32,
                                        scale=(0.08, 1),
                                        interpolation=InterpolationMode.BILINEAR,
                                        antialias=True),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]))

    grad_scaler = torch.cuda.amp.GradScaler()
    train_loader = DataLoaderX(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

    # load validation data
    val_dataset = torchvision.datasets.CIFAR10(root=args.val_dir, train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   normalize,
                                               ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('load data successfully')

    # load student model
    print("=> loading student model '{}'".format(args.model))
    model = ti_models.model_dict[args.model](num_classes=10)
    model = nn.DataParallel(model).cuda()
    model.train()
    ema_model = EMAMODEL(model)
    args.mode = "fkd_save"
    args.batch_size = args.batch_size // (args.gradient_accumulation_steps * ngpus_per_node)

    # aux_teacher = ["resnet18", "mobilenet_v2","shufflenet_v2_x0_5", "efficientnet_b0"]
    aux_teacher = ["ResNet18", "ConvNetW128", "MobileNetV2", "WRN_16_2", "ShuffleNetV2_0_5", "ConvNetD1","ConvNetD2","ConvNetW32"]
    args.aux_teacher = args.teacher_name = aux_teacher
    model_teacher = []
    for name in aux_teacher:
        if name in ["resnet18","mobilenet_v2","efficientnet_b0","shufflenet_v2_x0_5"]:
            _model = models.__dict__[name](pretrained=False, num_classes=10)
            if name == "resnet18":
                _model.conv1 = nn.Conv2d(
                    3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                _model.maxpool = nn.Identity()
                _model.fc = nn.Linear(_model.fc.in_features, 10)
            elif name == "mobilenet_v2":
                _model.features[0][0] = nn.Conv2d(
                    3, _model.features[0][0].out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                _model.classifier[1] = nn.Linear(_model.classifier[1].in_features,10)
            elif name == "efficientnet_b0":
                _model.features[0][0] = nn.Conv2d(
                    3, _model.features[0][0].out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                _model.classifier[1] = nn.Linear(_model.classifier[1].in_features,10)
            elif name == "shufflenet_v2_x0_5":
                _model.conv1 = nn.Conv2d(
                    3, _model.conv1[0].out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                _model.maxpool = nn.Identity()
                _model.fc = nn.Linear(_model.fc.in_features,10)
        else:
            if name in ["ConvNetW128","ConvNetD1", "ConvNetD2", "ConvNetW32"]:
                _model = ti_get_network(name, channel=3, num_classes=10, im_size=(32, 32), dist=False)
            else:
                _model = ti_models.model_dict[name](num_classes=10)
        model_teacher.append(_model)
        checkpoint = torch.load(
            os.path.join(args.pre_train_path, "CIFAR-10", name, f"squeeze_{name}.pth"),
            map_location="cpu")
        model_teacher[-1].load_state_dict(checkpoint)

    for _model in model_teacher:
        _model.cuda()
        _model = torch.nn.DataParallel(_model)    

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.ls_type == "cos":
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (args.st*args.epochs))) if step <= (args.st*args.epochs) else 0,
                             last_epoch=-1)
    elif args.ls_type == "cos2":
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (args.st*args.epochs))) if step <= (args.epochs*5/6) else 0.5 * (
                                     1. + math.cos(math.pi * 5 / (6 * args.st))) * (6*args.epochs-6*step)/(6*args.epochs),
                             last_epoch=-1)
    elif args.ls_type == "alrs":
        scheduler = ALRS(optimizer,decay_rate=args.alrs_dr)
    elif args.ls_type == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         [800,900,950],0.5)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0 - step / (args.st*args.epochs)) if step <= (args.st*args.epochs) else 0, last_epoch=-1)

    args.best_acc1 = 0 # 31.4% -> 34.4% (background)
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        now_loss = train(model, model_teacher, args, epoch, gpu, ngpus_per_node, scaler=grad_scaler, ema_model=ema_model)

        if (epoch % 10 == 0 and epoch > 800) or epoch == args.epochs - 1:
            ema_model.ema_swap(model)
            top1 = validate(model, args, epoch)
            ema_model.ema_swap(model)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        if args.ls_type == "alrs":
            scheduler.step(now_loss)
        else:
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, output_dir=args.output_dir)

    wandb.finish()


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, model_teacher, args, epoch=None, gpu=0, ngpus_per_node=1, scaler=None, ema_model=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    loss_function_dist = DISTLoss(tem=args.temperature)

    model.train()
    for _model in model_teacher:
        _model.eval()
    t1 = time.time()
    for batch_idx, (images, target) in enumerate(args.train_loader):
        images = images.cuda()
        target = target.cuda()
        images, _, _, _ = mix_aug(images, args)
        optimizer.zero_grad()
        soft_label = []

        with torch.no_grad() and torch.cuda.amp.autocast(enabled=True):
            for _model in model_teacher:
                soft_label.append(_model(images))
            soft_label = torch.stack(soft_label, 0)
            soft_label = soft_label.mean(0)

        with torch.cuda.amp.autocast(enabled=False):
            output = model(images).float()
            soft_label = soft_label.float()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        if args.loss_type == "kl":
            output = F.log_softmax(output / args.temperature, dim=1)
            soft_label = F.softmax(soft_label / args.temperature, dim=1)
            loss = loss_function_kl(output, soft_label)
        elif args.loss_type == "dist":
            loss = loss_function_dist(output, soft_label)
        elif args.loss_type == "mse_gt":
            loss = F.mse_loss(output, soft_label) + F.cross_entropy(output, target) * args.ce_weight
        elif args.loss_type == "mse_gt_sam":
            with torch.no_grad():
                ema_output = ema_model.ema_model(images).float()
            loss = F.mse_loss(output, soft_label) + F.cross_entropy(output, target) * args.ce_weight + \
                loss_function_kl(F.log_softmax(output, dim=1),F.softmax(ema_output, dim=1)) * 0.4
        else:
            raise NotImplementedError
        loss.backward()
        # scaler.scale(loss).backward()
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        if ema_model is not None:
            ema_model.ema_step(decay_rate=args.ema_dr,model=model)

        metrics = {
            "train/loss": objs.avg,
            "train/Top1": top1.avg,
            "train/Top5": top5.avg,
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch, }
        wandb_metrics.update(metrics)

    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()
    return objs.avg


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    main()
