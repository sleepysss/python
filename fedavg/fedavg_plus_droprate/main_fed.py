#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
import random
import math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__': #自己執行時才會做if下面的動作,被別人引用時則不會執行
    start = time.process_time()
    
    #print("pytorch gpu?",torch.has_mps)
    # parse args
    args = args_parser() #詳見python的Argparse
    #torch.device代表将torch.Tensor分配到的设备的对象，有cpu和cuda两种
    """
    if torch.cuda.is_available()
        device=torch.device("cuda:0")  只有“cuda”應該也可以
    else
        device=torch.device("cpu")
        
    torch.device("cuda:0") 指定使用第一個 CUDA 設備，而 torch.device("cuda") 則是使用預設的 CUDA 設備。
    如果系統上有多個 CUDA 設備，可以使用 "cuda:1"、"cuda:2" 等指定要使用的設備。如果只有一個 CUDA 設備，則兩
    者的效果是相同的。這個設定可以讓 PyTorch 在運行模型時使用 CUDA 設備進行加速運算。
    
    #agrs.gpu:-1的話是cpu,0則是gpu (詳見options.py)
    #.format() 是一個 Python 字符串格式化的方法
    #ex:
    #'Hello, {}!'.format('John') 中，字串中的 {} 會被 'John' 替換，最後生成的字串為 'Hello, John!'
    
    #多加一個屬性
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    """
    if torch.has_mps==True and args.gpu!=-1:
        args.device=torch.device("mps")  
        print("======using gpu(mps)======")
    elif torch.cuda.is_available() and args.gpu != -1:
        args.device=torch.device("cuda")
        print("======using gpu(cuda)======")
    else:
        args.device=torch.device("cpu")
        print("======using cpu======")
    
    # load dataset and split users
    if args.dataset == 'mnist':
        #括号内的数值可以被解析成一行
        #用transforms.Compose将transforms组合在一起。
        #ToTensor():將PIL Image或者 ndarray 轉換為tensor，並且歸一化至[0-1],並把(H,W,C)的矩阵转为(C,H,W)
        #因为pytorch很多函数都是设计成假设你的输入是 （c，h，w）的格式
        #HWC可以看作是一幅图像的shape，H表示图像的高度，W表示图像的宽度，而C表示一幅图像的通道数(channel)
        trans_mnist = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        #dataset_train[0]:type為tuple
        #img,label=dataset_train[0]  
        #img的shape為torch.Size([1,28,28]) shape可得到data的形狀
        #img的type為torch,Tensor, label的type為int
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users(分配哪些用戶得到dataset中的哪些照片)
        if args.iid:
            #dict_users為字典(int:set),dict_users[i]:用戶i分到的dataset中的img的index
            dict_users = mnist_iid(dataset_train, args.num_users) 
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True,transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True,transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    
    net_glob.train() #將模型從評估模式轉為訓練模式

    # copy weights
    w_glob = net_glob.state_dict()  #state_dict 會以字典的方式來儲存

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
    #應該是每個user都參加,即：fraction為1
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)] 
        
    local_ep=args.local_ep  
    for iter in range(args.epochs): #幾輪
        loss_locals = []
        if not args.all_clients: #if not:測試一個條件是否為假
            w_locals = [] #存m個model的weights
        m = max(int(args.frac * args.num_users), 1) #frac:比例(user中多少比例的可以參與訓練)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #選m個參與這輪
        #有些user無法完成全部local epochs
        exists_users = np.random.choice(idxs_users,int(math.ceil(m*(1-args.drop))),replace=False) 
        print("exists users:",exists_users.size)
        
        for idx in exists_users: #each user
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) #net=... 拿global model 
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals) #參與的user此輪的平均loss
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
    """
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid))
    """
    
    # testing
    print("======testing======")
    #训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，
    #否则的话，有输入数据，即使不训练，它也会改变权值。
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    
    end = time.process_time()
    print("執行時間：%f 秒" % (end - start))

