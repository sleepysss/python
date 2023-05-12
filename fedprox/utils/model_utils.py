import json
import numpy as np
import os
import re
import sys

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(models_dir, 'models')
sys.path.append(models_dir)

from client import Client

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [list], 'y': [list]}
    returns x, y, which are both lists of size-batch_size lists
    '''
    raw_x = data['x']
    raw_y = data['y']        
    batched_x = []
    batched_y = []
    for i in range(0, len(raw_x), batch_size):
        batched_x.append(raw_x[i:i+batch_size])
        batched_y.append(raw_y[i:i+batch_size])
    return batched_x, batched_y

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
      見data/mnist/generate_niid.py 已分好各user有的data了   
    
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')] #过滤出 train_files路徑中所有以 ".json" 结尾的文件名
    for f in train_files:
        file_path = os.path.join(train_data_dir,f) #完整地址
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
            
        """
        把 cdata['user_data'] 字典中的所有 key-value pair 添加到 train_data 字典中
      
        若cdata['user_data']的內容為
        {
            'f_00001': {'x': [[5, 6], [3, 4], [9, 10]], 'y': [0, 0, 0]},
            'f_00002': {'x': ...},
            'f_00003': {'x': ...}
        }
        則
        train_data最終會長成這樣
        {
            'f_00001': {'x': [[5, 6], [3, 4], [9, 10]], 'y': [0, 0, 0]},
            'f_00002': {'x': ...},
            'f_00003': {'x': ...}
        }
        """
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys()) #train_data 字典的所有 key 值

    return clients, groups, train_data, test_data #用tuple來傳回

def setup_clients(train_data_dir, test_data_dir, model=None):
    '''instantiates clients based on given train and test data directories

    Return:
        list of Clients
    '''
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    if len(groups) == 0:
        groups = [None for _ in users]
    all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return all_clients

