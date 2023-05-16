#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

#另一種資料分割的方法為用sampler
class DatasetSplit(Dataset):  #each user
    def __init__(self, dataset, idxs):
        #idxs:每個user分到的index(第幾張圖片),為一個set
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)
    
    #the __getitem__ function loads and return a sample from the dataset at the given index
    #item和sampler有關係
    def __getitem__(self, item): 
        image, label = self.dataset[self.idxs[item]]  #data sample
        return image, label


class LocalUpdate(object):   #each user   Python3中的類別都默認繼承自object,也可以class LocalUpdate:這樣寫
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):  #net:model
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep): #local端做幾輪
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad() #梯度清零
                log_probs = net(images) #前向傳播
                loss = self.loss_func(log_probs, labels) #計算損失函數
                loss.backward() #反向傳播
                optimizer.step() #更新local model的參數
                if self.args.verbose and batch_idx % 10 == 0: #我猜是local每10個batch印一次
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item()) #loss.item()可獲取當前的損失值
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) #local端此輪的loss
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) #傳回此user的weight和他平均的loss

