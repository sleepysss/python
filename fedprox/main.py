import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd', 'fedprox_origin']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist', 
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (26,),  # num_classes
    'mnist.mclr': (10,), # num_classes
    'mnist.cnn': (10,),  # num_classes
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic.mclr': (10, ) # num_classes
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='stacked_lstm.py')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--num_iters',
                        help='number of iterations when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)
    parser.add_argument('--mu',
                        help='constant for prox;',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--drop_percent',
                        help='percentage of slow devices',
                        type=float,
                        default=0.1)


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])
    
    print("model_path:",model_path)
    #导入指定路径下的 Python 模块
    #可以使用 learner 变量来实例化 Model 类的对象，并调用其方法。
    #Model:應該是flearn/models/mnist(可換其他)/mclr.py裡面的Model類
    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer(應該是指用flearn/trainers中的哪種方法(?))
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    print("opt_path:",opt_path)
    mod = importlib.import_module(opt_path) #動態導入
    optimizer = getattr(mod, 'Server')  #optimizer是個 flearn.trainers.[optimizer名稱].Server的class

    # add selected model parameter
    """
    這行程式碼的作用是從 model_path 中解析出模型的名稱，並從預先定義的 MODEL_PARAMS 字典中找到對應模型的參數，存儲在 
    parsed['model_params'] 中，以便在後續使用這些參數建立模型。
    假設 model_path 為 'flearn.models.mclr.model'，那麼 model_path.split('.')[2:] 就是 ['mclr', 'model']，
    接著用 '.'.join(model_path.split('.')[2:]) 把它們組合成模型名稱 'mclr.model'，最後用這個模型名稱從 MODEL_PARAMS 字典中
    獲取相應的超參數
    """
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]
    #print("parsed['model_params']:",parsed['model_params'])
    
    
    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    print("train_path:",train_path)
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    print("test_path:",test_path)
    dataset = read_data(train_path, test_path) #用tuple來接收

    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()
    
if __name__ == '__main__':
    main()
