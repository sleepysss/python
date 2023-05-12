import numpy as np
import tensorflow as tf
from tqdm import tqdm

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        
        # transfer parameters to self
        #这行代码的作用是将参数字典 params 中的每个键值对（即参数名和参数值）都设置为 Server 类的实例属性。
        #这样一来，在 Server 类中就可以直接通过 self.<参数名> 的方式来访问这些参数，而不必每次都通过
        #params['<参数名>'] 来访问它们
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        tf.reset_default_graph() #用這個函數來避免不同模型之間的變量名稱衝突
        #learner 变量指向 Model 类,下面這行的操作為使用 learner 变量来实例化 Model 类的对象
        #詳見main的read_options的learning
        #若 learner 是一個可實例化的類別 (class)，你可以直接使用以下語法進行實例化：
        #learner_instance = learner(param1, param2, ...)
        print("self.inner_opt",self.inner_opt) #印出來是有值的,我不知道她怎麼收到的==
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)  #self.inner_opt????
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            #在初始化時，每個列表元素都被賦值為 None。這表明尚未將任何用戶劃分到任何組中。
            groups = [None for _ in users]
        #all_clients 是一个包含多个 Client 对象的列表
        #每次迭代會從 users 和 groups 中分別取出一個元素 u 和 g，並把它們作為參數傳給 Client 類別的建構子，
        #建立一個 Client 物件，並把這個物件加入到 all_clients 這個 list 中
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients: 
            ct, cl, ns = c.train_error_and_loss() #不是recursive,是呼叫Client這個class的train_error_and_loss method
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        #返回值:第一個返回值是被選中客戶端的索引，第二個返回值是被選中客戶端對象的列表。
        #np.asarray(self.clients) 將 self.clients 轉換為 NumPy 的 array 型態，即將原本的 Python list 轉換成 NumPy array。
        #[indices]這個方法是 numpy 的 array indexing(indices是個ndarray,而不是常數)
        return indices, np.asarray(self.clients)[indices]
    
    #code長這樣的原因見fedavg(每個人的樣本數不同=>用加權平均)
    def aggregate(self, wsolns):
        total_weight = 0.0
        #wsolns[0][1]是一個很長的列表，其長度等於模型參數的總數。(詳見fedavg.py)
        base = [0]*len(wsolns[0][1]) #列表乘法（list multiplication） ex:10*[0] 會返回一個長度為 10 的列表
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

