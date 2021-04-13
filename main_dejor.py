from __future__ import print_function
import os
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
import random
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import *
import time
from Imagefolder_modified import Imagefolder_modified

# gpu = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu
#   python -u train_resnet18_.py --gama 0.1 --bs 50 2>&1 | tee bird_decouple_all_gama_0.1_bs_50.log
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='DeJoR')
parser.add_argument('--bs',  type=int, default=50,help='batch_size')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--net',  type=int, default=18,choices=[18,50],help='resnet18 or 50')
parser.add_argument('--data',  type=str, default='bird',help='dataset')
parser.add_argument('--gama',  type=float, default=2,help='coefficient of negative entropy term : gama')
parser.add_argument('--lamb', type=float, default=0.1, help ='coefficient of SCE : lambda')
parser.add_argument('--gpu', default='3', type=str, help='gpu_th')
parser.add_argument('--gpus', default=None, type=int, help='number of used gpu cards')
parser.add_argument('--save_dir',  type=str, default='bird' ,help='save dir')


args = parser.parse_args()
gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
args.gpus = len(gpu.split(','))
args.save_dir = args.data + '{:.2f}'.format(args.gama) + '_net_{}'.format(args.net)


class DejorLoss(nn.Module):
    r"""
    """
    def __init__(self, Tepoch =10, drop_rate = 0.25, class_num=200):
        super(DejorLoss, self).__init__()
        self.Tepoch = Tepoch
        self.drop_rate = drop_rate
        self.class_num = class_num


    def forward(self, logits_1, logits_2,labels, epoch):

        loss_sum, H = self.loss_sum_calculate(logits_1,logits_2,labels)

        ind_sorted = torch.argsort(loss_sum.data)  # (N) sorted index of the loss
        forget_rate = min(epoch, self.Tepoch)/self.Tepoch * self.drop_rate
        num_remember = math.ceil((1 - forget_rate) * logits_1.shape[0])
        ind_update = ind_sorted[:num_remember]  # select the first num_remember low-loss instances

        return loss_sum[ind_update].mean() - args.gama * H.mean()

    def loss_sum_calculate(self,logits_1,logits_2,labels):
        softmax1 = F.softmax(logits_1, dim=1)
        softmax2 = F.softmax(logits_2, dim=1)
        loss_1 = F.cross_entropy(logits_1, labels, reduction='none')  # (N) loss per instance in this batch
        loss_2 = F.cross_entropy(logits_2, labels, reduction='none')
        RCE = torch.sum(-torch.log(softmax1 + 1e-7) * softmax2, dim=-1) + \
              torch.sum(-torch.log(softmax2 + 1e-7) * softmax1, dim=-1)
        H = torch.sum(-torch.log(softmax1+ 1e-7) * softmax1, dim=-1) + \
              torch.sum(-torch.log(softmax2+ 1e-7) * softmax2, dim=-1)
        loss_sum = (1-args.lamb) * loss_1 + (1-args.lamb) * loss_2 + args.lamb * RCE         #  lambda=0.1

        return loss_sum, H


def train(nb_epoch, batch_size, store_name, start_epoch=0):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print('use cuda:', use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    trainset = Imagefolder_modified(root='./data/web-{}/train'.format(args.data), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
    print('train image number is ', len(trainset))
    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.ImageFolder(root='./data/web-{}/val'.format(args.data), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
    print('val image number is ', len(testset))

    # Model

    net1 = load_resnet_layer(model_name=args.net, classes_nums = len(trainset.classes), pretrain=True, require_grad=True)
    net2 = load_resnet_layer(model_name=args.net, classes_nums =len(trainset.classes),pretrain=True, require_grad=True)
    if args.gpus > 1:
        net1 = torch.nn.DataParallel(net1)
        net2 = torch.nn.DataParallel(net2)


    net1.cuda()
    net2.cuda()

    # CELoss = nn.CrossEntropyLoss()
    CoLoss = DejorLoss(class_num=len(trainset.classes))
    if args.gpus > 1:
        optimizer = optim.SGD([
            {'params': net1.module.parameters(), 'lr': args.lr},
            {'params': net2.module.parameters(), 'lr': args.lr}
        ], momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.SGD([
            {'params': net1.parameters(), 'lr': args.lr},
            {'params': net2.parameters(), 'lr': args.lr}
        ], momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                              # if acc is not max for patience step, set lr=lr*factor
                                                              patience=3, verbose=True, threshold=1e-4)
    if os.path.exists(exp_dir + '/results_train.txt'):
        os.remove(exp_dir + '/results_train.txt')
    if os.path.exists(exp_dir + '/results_test.txt'):
        os.remove(exp_dir + '/results_test.txt')
    if os.path.exists(exp_dir + '/acc.txt'):
        os.remove(exp_dir + '/acc.txt')

    max_val_acc_com = 0
    max_val_acc_bet = 0
    max_com_epoch = 0
    # lr = [0.001,0.001]
    for epoch in range(start_epoch, nb_epoch):

        start = time.time()
        net1.train()
        net2.train()
        train_loss = 0
        correct = 0
        total = 0

        for idx, (inputs, targets, index) in enumerate(trainloader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            output_1 = net1(inputs)
            output_2 = net2(inputs)

            loss = CoLoss(output_1,output_2, targets, epoch)                 #   (self, logits, targets, index, epoch, layer_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted1 = torch.max(output_1.data, 1)
            _, predicted2 = torch.max(output_2.data, 1)
            total += targets.size(0)
            correct += (predicted1.eq(targets.data).cpu().sum()+predicted2.eq(targets.data).cpu().sum())/2.
            train_loss += loss.item()

        train_acc = 100. * float(correct) / total

        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write('Iteration %d | train_acc = %.5f | train_loss = %.5f | | time%.1f min(%.1fh)\n' % (
                epoch, train_acc, train_loss/ (idx + 1), (time.time()-start)/60,(time.time()-start)*(nb_epoch-epoch-1)/3600 ))


        if epoch >= 0:
            net1.eval()
            net2.eval()
            top1_val = AverageMeter()
            top2_val = AverageMeter()
            topcom_val = AverageMeter()
            idx = 0

            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(testloader):
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    output_1 = net1(inputs)
                    output_2 = net2(inputs)
                    outputs_com = output_1 + output_2

                    prec1 = accuracy(outputs_com.float().data, targets)[0]
                    topcom_val.update(prec1.item(), inputs.size(0))

                    prec1 = accuracy(output_1.float().data, targets)[0]
                    top1_val.update(prec1.item(), inputs.size(0))
                    prec1 = accuracy(output_2.float().data, targets)[0]
                    top2_val.update(prec1.item(), inputs.size(0))


            prec1 = top1_val.avg
            prec2 = top2_val.avg
            preccom = topcom_val.avg
            precbet = max(prec1,prec2)

            val_com = preccom
            val_bet = precbet

            fw = open(exp_dir + "/acc.txt", 'a')
            fw.write('{:4.3f} {:4.3f} {:4.3f} {:4.3f}\n'.format(prec1,prec2,val_bet,val_com))
            fw.close()

            show_param = 'epoch: %d | sum Loss: %.3f | train Acc: %.3f%%  | test bet Acc: %.3f%% comacc: %.3f%% | time%.1f min(%.1fh)' % (
                    epoch,  train_loss/ (idx + 1), train_acc, val_bet, val_com, (time.time()-start)/60, (time.time()-start)*(nb_epoch-epoch-1)/3600 )
            if val_com > max_val_acc_com:
                max_val_acc_com = val_com
                max_com_epoch = epoch
                print('*'+show_param)
            else:
                print(show_param)

            if val_bet > max_val_acc_bet:
                max_val_acc_bet = val_bet

            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_bet = %.5f, test_acc_comb = %.5f\n' % (
                epoch, val_bet,val_com))
        lr_scheduler.step(val_bet)
    print('--------------------------------------------')
    print('best val acc: {}(bet) {}(comb), com epoch {}, lambda {} gama{:.1f} bs {}'.format(max_val_acc_bet, max_val_acc_com, max_com_epoch,args.lamb,args.gama,args.bs))
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('best val acc: {}(bet) {}(comb), com epoch {}, lambda {} gama{:.1f} bs {}'.format(max_val_acc_bet, max_val_acc_com, max_com_epoch,args.lamb,args.gama,args.bs))

start_time = time.time()
train(nb_epoch=100,             # number of epoch
         batch_size=args.bs,#110,         # batch size
         store_name=args.save_dir,     # folder for output
         start_epoch=0,         # the start epoch number when you resume the training
) 

print('--------------------------------------------')
print('total time: {}h'.format((time.time()-start_time)//3600))


