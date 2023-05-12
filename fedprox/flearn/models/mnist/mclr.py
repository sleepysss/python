import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''
    #optimizer:tf.train.GradientDescentOptimizer類的object或作者自己定義的optimizer(flearn/optimizer)的object,但作者自己定義的也是繼承來的
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        self.num_classes = num_classes

        # create computation graph        
        self.graph = tf.Graph() #tf.Graph是计算图的静态描述，包含计算图中的所有操作和变量
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            #計算模型的浮點運算次數
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        #构建全连接层的快捷方法 tf.layers.dense
        #邏輯回歸模型中只有一層網絡，它是一個線性模型。它的輸入為特徵向量，輸出為標籤的預測概率
        logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss) #返回一个元组列表，其中每个元组包含一个梯度和对应的变量。
        grads, _ = zip(*grads_and_vars)
        #train_op代表了執行optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())這個操作
        #(我也不知道這啥鬼笑死,而不是返回值)
        #这个东西的专有名词是“Operation”，在TensorFlow中被定义为计算图中的节点，可以理解为计算图中的一个操作或指令。
        #每个Operation可以接受输入，可以产生输出，也可能不产生输出。一般情况下，我们在TensorFlow中定义了一些操作后，
        #需要将它们组成一个计算图，并通过会话(Session)来执行这个计算图中的操作。
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        #計算模型評估指標的操作
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)
                    
    #获取模型的可训练变量参数（trainable variables）
    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads
    
    #train
    #单个客户端上训练模型的过程
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120): #顯示一個進度條,迭代num_epochs次的訓練過程
            #每次迭代会返回一个批次的数据
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    #执行前向传播和反向传播，并更新网络的参数
                    #X,y是真的數據
                    self.sess.run(self.train_op,feed_dict={self.features: X, self.labels: y})  
        soln = self.get_params() #得到模型的參數
        #計算了整個訓練過程中所執行的浮點運算次數 (FLOPs) 的總數
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp
    
    #在一个客户端上多次迭代训练模型的过程(和呼叫多次solve_inner的效果可能類似)
    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = 0
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
