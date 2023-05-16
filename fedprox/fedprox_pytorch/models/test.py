#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            #data, target = data.cuda(), target.cuda()
            #.to(device) 可以指定CPU 或者GPU
            #.cuda() 只能指定GPU
            data, target = data.to(args.device), target.to(args.device) #mac的叫mps不是cuda
        log_probs = net_g(data) #前向傳播
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()  #sum:總損失
        # get the index of the max log-probability
        #返回log_probs中每个样本对应的最大概率值和最大概率值所在的索引
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        #若預測的答案和target相同，correct就累加1
        """
        連續調用method
        y = x.sum().item()
        等同
        y_sum = x.sum()
        y_item = y_sum.item()
        """
        # 沒加.cpu()會有warning
        correct += y_pred.eq(target.data.view_as(y_pred)).cpu().long().sum() 

    test_loss /= len(data_loader.dataset)  #dataset中样本的数量
    accuracy = 100.00 * correct / len(data_loader.dataset) #percent
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

