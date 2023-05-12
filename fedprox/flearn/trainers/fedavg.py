import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate']) #fedavg的優化器為SGD
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Avging'''
        #clients_per_round是argparse的一個參數,在BaseFederated有把全部的參數換成可用base呼叫
        print('Training with {} workers ---'.format(self.clients_per_round))  

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:  #eval_every:evaluate every ____ rounds,定義在argparse參數中
                stats = self.test()  # have set the latest model for all clients,定義在BaseFederated
                stats_train = self.train_error_and_loss() #定義在BaseFederated

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
            
            #selected_clients是ndarray,array中的每個元素是個Client類的object,代表每一user,詳見BaseFederated
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i) #設定隨機種子，使得在執行隨機過程時每次得到的結果都是固定的。
            #我猜可能是選取幾個能在時間內完成的user
            #actie_clients:
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  # buffer for receiving client solutions(list of tuple)
            
            #local端做training
            #c是Client類的object
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally(local端做training)
                #詳見client.py的solve_inner的回傳值:(self.num_samples, soln), (bytes_w, comp, bytes_r)
                #1: num_samples: number of samples used in training
                #1: soln: local optimization solution
                #2: bytes read: number of bytes received
                #2: comp: number of FLOPs executed in training process
                #2: bytes_write: number of bytes transmitted
                #所以soln和stats都是tuple,包含以上這些東西
                
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size) 

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            #回傳是model weights(type:list)
            self.latest_model = self.aggregate(csolns) #定義在BaseFederated

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
