import argparse
import glob
import os
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pytorch_lightning as pl

from multimodal.utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Linear decoding with headcam data')
parser.add_argument('--train_dir', metavar='DIR', help='path to train dataset')
parser.add_argument('--split', type=str, default='first', choices=['first', 'last'],
                    help='split to use for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1024), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--num-classes', default=64, type=int, help='number of classes in downstream classification task')
parser.add_argument('--seed', type=int, default=0, help='random seed')

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def load_split_train_test(train_dir, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    if args.split == "first":
        # split dataset so first half of each class is used for training
        train_indices = []
        test_indices = []
        for i in range(len(train_data.classes)):
            indices = [j for j, x in enumerate(train_data.targets) if x == i]
            train_split_indices = indices[:int(len(indices) * 0.5)]
            test_split_indices = indices[int(len(indices) * 0.5):]
            train_indices += train_split_indices
            test_indices += test_split_indices
    elif args.split == "last":
        # split dataset so last half of each class is used for training
        train_indices = []
        test_indices = []
        for i in range(len(train_data.classes)):
            indices = [j for j, x in enumerate(train_data.targets) if x == i]
            train_split_indices = indices[int(len(indices) * 0.5):]
            test_split_indices = indices[:int(len(indices) * 0.5)]
            train_indices += train_split_indices
            test_indices += test_split_indices

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, shuffle=False)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True, shuffle=False)

    num_train = len(train_loader)
    num_test = len(test_loader)
    print('Total train data size is', num_train * args.batch_size)
    print('Total test data size is', num_test * args.batch_size)

    return train_loader, test_loader

def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    num_classes = args.num_classes

    # set random seed
    pl.seed_everything(args.seed)

    # load model
    model_name = "dino_sfp_resnext50"
    model = load_model(model_name, pretrained=True)
    model = model.to(device)
    
    set_parameter_requires_grad(model) 
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.num_classes, bias=True).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Data loading code
    savefile_name = f'probe_results/object_categories_self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_seed_{args.seed}_split_{args.split}.tar'

    train_loader, test_loader = load_split_train_test(args.train_dir, args)
    acc1_list = []
    val_acc1_list = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)

    # validate at end of epoch
    val_acc1, preds, target, images = validate(test_loader, model, args)
    val_acc1_list.append(val_acc1)

    torch.save({'acc1_list': acc1_list,
                'val_acc1_list': val_acc1_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'preds': preds,
                'target': target,
                'images': images
                }, savefile_name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for param in model.parameters():
        #     print(param.requires_grad)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg.cpu().numpy()


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            preds = np.argmax(output.cpu().numpy(), axis=1)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg, preds, target.cpu().numpy(), images.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
            
