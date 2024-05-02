import os,math
import argparse
import sys

sys.path.append("../")
import torch
import torchvision
from torchvision import transforms
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import ImageNetwithClass
import torchvision.models as models

class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=4, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.cel_reduction = 'mean' if reduction == 'batchmean' else reduction

    def forward(self, student_output, teacher_output, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        return (self.temperature ** 2) * soft_loss


class EMAMODEL(object):
    def __init__(self, model):
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self, model=None, decay_rate=0.999):
        for param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)

    @torch.no_grad()
    def ema_swap(self, model=None):
        for param, ema_param in zip(self.ema_model.parameters(), model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp

    @torch.no_grad()
    def __call__(self, x):
        return self.ema_model(x)


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    traindataset = ImageNetwithClass(root=args.data_path, nclass=10,
                                          transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]))
    trainloader = torch.utils.data.DataLoader(traindataset,
                                num_workers=4,
                                batch_size=args.batch_size,
                                drop_last=False,
                                shuffle=True)


    model = models.__dict__[args.model](pretrained=True, num_classes=1000)
    if args.model == "resnet18":
        model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features,10)
    elif args.model == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,10)
    elif args.model == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,10)
    elif args.model == "shufflenet_v2_x0_5":
        model.fc = nn.Linear(model.fc.in_features,10)
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.squeeze_path, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ''' organize the real dataset '''

    criterion = nn.CrossEntropyLoss().to(args.device)
    model = model.to(args.device)
    model.train()
    lr = args.lr_teacher
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: 0.5 * (
                                                          1. + math.cos(
                                                      math.pi * step / (args.train_epochs*3))) if step <= args.train_epochs else 0,
                                                  last_epoch=-1)
    for e in range(args.train_epochs):
        total_acc = 0
        total_number = 0
        model.train()

        for batch_idx, (input, target) in enumerate(trainloader):
            input = input.float().cuda()
            target = target.cuda()
            target = target.view(-1)
            optimizer.zero_grad()
            logit = model(input)

            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()

            total_acc += (target == logit.argmax(1)).float().sum().item()
            total_number += target.shape[0]

        scheduler.step()

        top_1_acc = round(total_acc * 100 / total_number, 3)
        print(f"Epoch: {e}, Top-1 Accuracy: {top_1_acc}%")
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, f"squeeze_{args.model}.pth"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='ImageNet-10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--squeeze_path', type=str, default='./squeeze_wo_ema/', help='squeeze path')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    args = parser.parse_args()
    main(args)
