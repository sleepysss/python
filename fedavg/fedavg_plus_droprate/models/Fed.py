#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w): #w:含有m個model的weights (list of OrderedDict)
    w_avg = copy.deepcopy(w[0]) #第一個local model的weight,此weight是用OrderedDict存的
    
    #key：layer 的名稱、value：參數值
    for k in w_avg.keys(): #model中的各個layer  ,字典(Dictionary)的keys()函数以列表返回一个字典所有的键
        for i in range(1, len(w)): #m-1個user
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)) #model的第k層(layer)的平均的結果
    return w_avg
