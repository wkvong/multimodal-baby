import argparse
import glob
import os
import random
import shutil
import time
import warnings
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

from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Linear decoding with headcam data')
parser.add_argument('--train_dir', metavar='DIR', help='path to train dataset')
parser.add_argument('--test_dir', metavar='DIR', help='path to test dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1024), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
# parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
parser.add_argument('--num-classes', default=22, type=int, help='number of classes in downstream classification task')
parser.add_argument('--subset', default=100, type=int, choices=[1, 10, 100],
                    help="proportion of training data to use for linear probe")

def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_split_train_test(train_dir, test_dir, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data = datasets.ImageFolder(test_dir, transform=transforms.Compose([transforms.ToTensor(), normalize]))

    # create subsets and setup dataloaders
    if args.subset == 100:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    if args.subset == 10:
        # create subset with 10% of training data using SubsetRandomSampler
        train_indices = list(range(len(train_data)))
        random.shuffle(train_indices)
        train_indices = train_indices[:int(0.1 * len(train_indices))]
        print(len(train_indices))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, shuffle=False)
    elif args.subset == 1:
        # create subset with 1% of training data
        train_indices = list(range(len(train_data)))
        random.shuffle(train_indices)
        train_indices = train_indices[:int(0.01 * len(train_indices))]
        print(len(train_indices))
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    num_train = len(train_loader)
    num_test = len(test_loader)
    print('Total train data size is', num_train * args.batch_size)
    print('Total test data size is', num_test * args.batch_size)

    return train_loader, test_loader

def main():
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    num_classes = args.num_classes

    # load our model checkpoint
    checkpoint_name = args.checkpoint
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]

    # get seed
    if 'seed_0' in checkpoint_name:
        seed = 0
    elif 'seed_1' in checkpoint_name:
        seed = 1
    elif 'seed_2' in checkpoint_name:
        seed = 2

    model = MultiModalLitModel.load_from_checkpoint(
        checkpoint, map_location=device)
    vision_model = model.vision_encoder

    # define custom vision model so that only first return arg from forward pass is used
    # which contains the actual embedding
    class VisionModelWrapper(nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        def forward(self, x):
            x, y = self.vision_model(x)
            return x

    vision_model = VisionModelWrapper(vision_model)
    set_parameter_requires_grad(vision_model)
    classifier = torch.nn.Linear(in_features=512, out_features=args.num_classes, bias=True).to(device)
    model = torch.nn.Sequential(vision_model, classifier)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Data loading code
    if 'finetune_cnn_True' in checkpoint_name:
        savefile_name = f'probe_results/embedding_finetuned_pretrained_contrastive_labeled_s_linear_probe_seed_{seed}_subset_{args.subset}.tar'
    else:
        savefile_name = f'probe_results/embedding_frozen_pretrained_contrastive_labeled_s_linear_probe_seed_{seed}_subset_{args.subset}.tar'

    train_loader, test_loader = load_split_train_test(args.train_dir, args.test_dir, args)
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
