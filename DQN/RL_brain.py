import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q network off-policy

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features, # 做什么用的？
        learning_rate=0.01,
        reward_decay=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features # TODO: n_features的含义是？
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max # TODO: 这一步为什么这样赋值？

        # total learning step
        self.learn_step_counter = 0

        # 初始化[s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        # tf.get_collection()：从一个集合中取出去全部变量，是一个列表。通常与tf.add_to_collection()结合使用。这相当于提供了一个全局的存储机制，不会受到变量名生存空间的影响
        # Q:这里对应的add_to_collection()在哪？
        # A:在_build_net中调用tf.get_variable()方法中，会自动将变量添加至tf.GraphKeys.GLOBAL_VARIABLES或者给定的collection中(例如eval_net_params)
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        # tf.assgin(t, e)将t的值变为e
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if(output_graph):
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter('logs/', self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------ build evaluate_net -----
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target') # for calculate loss

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 在更新target_net参数时会用到
            # n_l1是第一层神经元的数量，w_initializer, b_initializer分别是weight和bias的初始化值
            # TODO: 为什么w和b的初始化值只有单个值？不应该是矩阵和向量吗？
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # eval_net第一层，collections在更新target_net参数时用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names) # collections属性是一个graph collections keys的列表，用于添加变量至对应名字下的collection中
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relt(tf.matmul(self.s, w1) + b1)

            # eval_net第二层，collections在更新target_net参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.Variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ----- build target_net -----
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一层,c_names(collections_names)在更新target_net参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            
            # 第一层,c_names(collections_names)在更新target_net参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=w_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # np.hstack() horison方向堆叠，即合并成一行
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have
