#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users: number of users
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)  #len(dataset): dataset中有幾張照片,所以num_item應該是一個user幾張
    #dict_users是個字典 key:int,value:set
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] #多重指定,list comprehension
    for i in range(num_users):
        #从数组中随机抽取元素
        #np.random.choice()从a(一维数据)中随机抽取数字，返回指定大小(size)的数组(type為ndarray)
        #ex:
        #arr = np.array([1, 2, 3, 2, 1])
        # 將 ndarray 轉換成 set
        #s = set(arr)
        #print(s)  # 輸出結果：{1, 2, 3}
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) #每個user分到的index
        all_idxs = list(set(all_idxs) - dict_users[i]) #被分走的就從裡面移出,不然可能不同user會拿到同一筆
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300 #分成200個碎片,每一個碎片有300張img
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs) #此陣列以等差數列的形式產生 np.arange(3) => [0,1,2]
    labels = dataset.targets.numpy() #將label轉成numpy array  原本：dataset.train_labels.numpy()但有warning

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  #對2-d array進行排序（排序標準：label）
    idxs = idxs_labels[0,:] #見numpy的slice,這個操作會取得第0列和全部的行（即完整的第0列）
    
    """
    ex:
    idxs = [2, 0, 3, 1]
    labels = [1, 2, 0, 2]
    idxs_labels=[[0 1 2 3]
                 [1 2 0 2]]
    idxs_labels=[[1 3 0 2]
                 [0 1 2 2]]
    idxs=[1 3 0 2]
    """

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False)) #全部碎片中的其中兩個
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            #np.concatenate:合併兩個陣列
            #ex:
            #x = np.array([1, 2, 3]) =>[1, 2, 3] 
            #y = np.array([4, 5, 6]) =>[4, 5, 6]
            #np.concatenate([x, y]) =>array([1, 2, 3, 4, 5, 6])
            #rand*num_imgs:(rand+1)*num_imgs:該碎片的“勢力範圍”
            #ex:
            #rand=0
            #0*300,1*300=>0~300(300不含)
            #且因為排過了,對應的label的總類最多為2（應該吧）,大部分應該都是同一種label
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
