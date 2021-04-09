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

# gpu = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu
#    python -u train_pmg_sce_sumloss_decouple_sl.py --gama 0 --bs 30 --gpu 3,4 --net 50 --intro 34card_gama_0 2>&1 | tee bird_decouple_all_gama_0_bs_30_pmgnet50.log
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Multi-Scale DeJoR')
parser.add_argument('--bs',  type=int, default=50,help='batch_size')
parser.add_argument('--net',  type=int, default=18,choices=[18,50,16],help='MS resnet18, resnet50 or vgg16')
parser.add_argument('--data',  type=str, default='bird',help='dataset')
parser.add_argument('--gama',  type=float, default=2,help='coefficient of negative entropy term : gama')
parser.add_argument('--lamb', type=float, default=0.1, help ='coefficient of SCE : lambda')
parser.add_argument('--gpu', default='1,3', type=str, help='gpu_th')
parser.add_argument('--gpus', default=None, type=int, help='number of used gpu cards')
parser.add_argument('--save_dir',  type=str, default='bird' ,help='save dir')


args = parser.parse_args()
gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
args.gpus = len(gpu.split(','))
print(args.gpu, 'total cards:', args.gpus)
args.save_dir = args.data + '{:.2f}_ms_{}'.format(args.gama,args.net)


class MsdejorLoss(nn.Module):
    r"""
    """
    def __init__(self, Tepoch =10, drop_rate = 0.25, class_num=200):
        super(MsdejorLoss, self).__init__()
        self.Tepoch = Tepoch
        self.drop_rate = drop_rate

        self.class_num = class_num


    def forward(self, logits_1, logits_2, logits_3, logits_4,logits_5, logits_6,logits_7, logits_8,labels, epoch):

        sloss_sum1, hloss1 = self.loss_sum_calculate(logits_1,logits_2,labels)
        sloss_sum2, hloss2 = self.loss_sum_calculate(logits_3,logits_4,labels)
        sloss_sum3, hloss3 = self.loss_sum_calculate(logits_5,logits_6,labels)
        sloss_sum4, hloss4 = self.loss_sum_calculate(logits_7,logits_8,labels)
        sloss_sum = sloss_sum1 + sloss_sum2 + sloss_sum3 + 2 * sloss_sum4
        hloss = hloss1 + hloss2 + hloss3 + 2 * hloss4
        ind_sorted = torch.argsort(sloss_sum.data)  # (N) sorted index of the loss

        forget_rate = min(epoch, self.Tepoch)/self.Tepoch * self.drop_rate
        num_remember = math.ceil((1 - forget_rate) * logits_1.shape[0])
        ind_update = ind_sorted[:num_remember]  # select the first num_remember low-loss instances

        return sloss_sum[ind_update].mean() - args.gama * hloss.mean()

    def loss_sum_calculate(self,logits_1,logits_2,labels):
        softmax1 = F.softmax(logits_1, dim=1)
        softmax2 = F.softmax(logits_2, dim=1)
        loss_1 = F.cross_entropy(logits_1, labels, reduction='none')  # (N) loss per instance in this batch
        loss_2 = F.cross_entropy(logits_2, labels, reduction='none')
        RCE = torch.sum(-torch.log(softmax1 + 1e-7) * softmax2, dim=-1) + \
              torch.sum(-torch.log(softmax2 + 1e-7) * softmax1, dim=-1)
        H = torch.sum(-torch.log(softmax1 + 1e-7) * softmax1, dim=-1) + \
              torch.sum(-torch.log(softmax2 + 1e-7) * softmax2, dim=-1)

        sloss_sum = (1-args.lamb) * loss_1 + (1-args.lamb) * loss_2 + args.lamb * RCE         #  lambda=0.1

        return sloss_sum, H

def train(nb_epoch, batch_size, store_name, start_epoch=0):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print('use cuda:',use_cuda)

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

    if args.net == 18:
        net1 = load_ms_layer(model_name='resnet18_ms',classes_nums=len(trainset.classes), pretrain=True, require_grad=True)
        net2 = load_ms_layer(model_name='resnet18_ms',classes_nums=len(trainset.classes), pretrain=True, require_grad=True)
    elif args.net == 50:
        net1 = load_ms_layer(model_name='resnet50_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
        net2 = load_ms_layer(model_name='resnet50_ms', classes_nums=len(trainset.classes),pretrain=True, require_grad=True)
    if args.gpus > 1:
        net1 = torch.nn.DataParallel(net1)
        net2 = torch.nn.DataParallel(net2)

    net1.cuda()
    net2.cuda()

    CoLoss = MsdejorLoss(class_num=len(trainset.classes))
    if args.gpus > 1:
        optimizer = optim.SGD([
            {'params': net1.module.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block1.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier1.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block2.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier2.parameters(), 'lr': 0.002},
            {'params': net1.module.conv_block3.parameters(), 'lr': 0.002},
            {'params': net1.module.classifier3.parameters(), 'lr': 0.002},
            {'params': net1.module.features.parameters(), 'lr': 0.0002},

            {'params': net2.module.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block1.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier1.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block2.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier2.parameters(), 'lr': 0.002},
            {'params': net2.module.conv_block3.parameters(), 'lr': 0.002},
            {'params': net2.module.classifier3.parameters(), 'lr': 0.002},
            {'params': net2.module.features.parameters(), 'lr': 0.0002}
        ], momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = optim.SGD([
            {'params': net1.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net1.conv_block1.parameters(), 'lr': 0.002},
            {'params': net1.classifier1.parameters(), 'lr': 0.002},
            {'params': net1.conv_block2.parameters(), 'lr': 0.002},
            {'params': net1.classifier2.parameters(), 'lr': 0.002},
            {'params': net1.conv_block3.parameters(), 'lr': 0.002},
            {'params': net1.classifier3.parameters(), 'lr': 0.002},
            {'params': net1.features.parameters(), 'lr': 0.0002},

            {'params': net2.classifier_concat.parameters(), 'lr': 0.002},
            {'params': net2.conv_block1.parameters(), 'lr': 0.002},
            {'params': net2.classifier1.parameters(), 'lr': 0.002},
            {'params': net2.conv_block2.parameters(), 'lr': 0.002},
            {'params': net2.classifier2.parameters(), 'lr': 0.002},
            {'params': net2.conv_block3.parameters(), 'lr': 0.002},
            {'params': net2.classifier3.parameters(), 'lr': 0.002},
            {'params': net2.features.parameters(), 'lr': 0.0002}
        ], momentum=0.9, weight_decay=1e-5)

    if os.path.exists(exp_dir + '/ms_results_train.txt'):
        os.remove(exp_dir + '/ms_results_train.txt')
    if os.path.exists(exp_dir + '/ms_results_test.txt'):
        os.remove(exp_dir + '/ms_results_test.txt')
    if os.path.exists(exp_dir + '/ms_acc.txt'):
        os.remove(exp_dir + '/ms_acc.txt')

    max_val_acc_com = 0
    max_val_acc = 0
    max_val_acc2=0
    max_com_epoch = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]*2
    for epoch in range(start_epoch, nb_epoch):
        # print('Epoch: %d' % epoch)
        start = time.time()
        net1.train()
        net2.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0

        for batch_idx, (inputs, targets, index) in enumerate(trainloader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            output_1_1, output_2_1, output_3_1, output_concat_1 = net1(inputs, 4)
            output_1_2, output_2_2, output_3_2, output_concat_2 = net2(inputs, 4)

            loss = CoLoss(output_1_1,output_1_2, output_2_1, output_2_2,output_3_1,output_3_2,output_concat_1,output_concat_2, targets, epoch)                 #   (self, logits, targets, index, epoch, layer_number)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #  training log
            _, predicted1 = torch.max(output_concat_1.data, 1)
            _, predicted2 = torch.max(output_concat_2.data, 1)
            total += targets.size(0)
            correct += (predicted1.eq(targets.data).cpu().sum()+predicted2.eq(targets.data).cpu().sum())/2.

            train_loss += loss.item()

            # print(loss1_1.item(),loss2_1.item(),loss2_1.item(),concat_loss_1.item(), (loss1_1.item() + loss2_1.item() + loss3_1.item() + concat_loss_1.item() ),batch_idx)

        train_acc = 100. * float(correct) / total

        with open(exp_dir + '/ms_results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, train_acc, train_loss/ (idx + 1) ))

        # if epoch < 20 or epoch >= 80:
        # if epoch <80:
        #     print('epoch: %d | sum Loss: %.3f | train Acc: %.3f%%  | time%.1f min(%.1fh)' % (
        #             epoch,  train_loss/ (idx + 1), train_acc, (time.time()-start)/60, (time.time()-start)*(nb_epoch-epoch-1)/3600 ))
        if epoch >= 0:
            net1.eval()
            net2.eval()

            topconcat_1_val = AverageMeter()
            topconcat_2_val = AverageMeter()
            topcom_1_val = AverageMeter()
            topcom_2_val = AverageMeter()
            topcom_val = AverageMeter()
            total = 0
            idx = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    idx = batch_idx
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    output_1_1, output_2_1, output_3_1, output_concat_1 = net1(inputs, 4)
                    output_1_2, output_2_2, output_3_2, output_concat_2 = net2(inputs, 4)
                    outputs_com_1 = output_1_1 + output_2_1 + output_3_1 + output_concat_1
                    outputs_com_2 = output_1_2 + output_2_2 + output_3_2 + output_concat_2

                    outputs_com = outputs_com_1 + outputs_com_2
                    prec1 = accuracy(outputs_com.float().data, targets)[0]
                    topcom_val.update(prec1.item(), inputs.size(0))

                    prec1 = accuracy(output_concat_1.float().data, targets)[0]
                    topconcat_1_val.update(prec1.item(), inputs.size(0))
                    prec1 = accuracy(output_concat_2.float().data, targets)[0]
                    topconcat_2_val.update(prec1.item(), inputs.size(0))

                    prec1 = accuracy(outputs_com_1.float().data, targets)[0]
                    topcom_1_val.update(prec1.item(), inputs.size(0))
                    prec1 = accuracy(outputs_com_2.float().data, targets)[0]
                    topcom_2_val.update(prec1.item(), inputs.size(0))

            precconcat_1 = topconcat_1_val.avg
            precconcat_2 = topconcat_2_val.avg
            preccom_1 = topcom_1_val.avg
            preccom_2 = topcom_2_val.avg

            correctconcat = max(precconcat_1,precconcat_2)
            correctcom = max(preccom_1, preccom_2)


            val_acc = correctconcat
            val_acc_com = correctcom
            val_acc_2 = topcom_val.avg
            # print(correct_val_concat, correct_val_com, val_acc, val_acc_com)
            fw = open(exp_dir + "/ms_acc.txt", 'a')
            fw.write('{:4.3f} {:4.3f}'
                     ' {:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f}\n'.format(precconcat_1,precconcat_2,preccom_1,preccom_2,
                                                                  correctconcat,correctcom,val_acc_2))
            fw.close()


            show_param = 'epoch: %d | gama %.2f |sum Loss: %.3f | train Acc: %.3f%%  | test Acc: %.3f%% comacc: %.3f%% comacc2: %.3f%% | time%.1fmin(%.1fh)' % (
                    epoch, args.gama, train_loss/ (idx + 1),
                    train_acc,val_acc, val_acc_com,val_acc_2, (time.time()-start)/60, (time.time()-start)*(nb_epoch-epoch-1)/3600 )
            if val_acc_com > max_val_acc_com:
                max_val_acc_com = val_acc_com
                max_com_epoch = epoch
                print('*'+show_param)
            else:
                print(show_param)

            if val_acc > max_val_acc:
                max_val_acc = val_acc
            if val_acc_2 > max_val_acc2:
                max_val_acc2 = val_acc_2

            with open(exp_dir + '/ms_results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_acc_combined2 = %.5f\n' % (
                epoch, val_acc, val_acc_com,val_acc_2))

    print('--------------------------------------------')
    print('best val acc: {} {}(comb) {}(comb2), com epoch {}, lambda {} gama {} bs {}'.format(max_val_acc,max_val_acc_com,max_val_acc2,max_com_epoch,args.lamb,args.gama,args.bs))
    with open(exp_dir + '/ms_results_test.txt', 'a') as file:
        file.write('best val acc: {} {}(comb) {}(comb2), com epoch {}, lambda {}  gama {} bs {}'.format(max_val_acc,max_val_acc_com,max_val_acc2,max_com_epoch,args.lamb,args.gama,args.bs))

start_time = time.time()
train(nb_epoch=100,             # number of epoch
         batch_size=args.bs,         # batch size
         store_name=args.save_dir,     # folder for output
         start_epoch=0,         # the start epoch number when you resume the training
)       

print('--------------------------------------------')
print('total time: {}h'.format((time.time()-start_time)//3600))


