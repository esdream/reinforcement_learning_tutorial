import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space # action_space是一个list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if(state not in self.q_table.index):
            # dataframe.append()将series或dataframe添加到末尾
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if(np.random.rand() < self.epsilon):
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass
        

# on-policy learning
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # super函数是用用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。总之前人留下的经验就是：保持一致性。要不全部用类名调用父类，要不就全部用 super，不要一半一半。
        # 用法如下super(本类名, self).__init__()
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if(s_ != 'terminal'):
            # 选择采取下一个行动的state和action
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # next state is not terminal
        else:
            q_target = r # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

# backward eligibility traces
class SarsaLambdaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()
        
    def check_state_exist(self, state):
        if(state not in self.q_table.index):
            # append new state to q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
    
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if(s_ != 'terminal'):
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # next state is not terminal
        else:
            q_target = r # next state is terminal
        error = q_target - q_predict

        # 对于经历过的state-action, 我们让他+1, 证明它是得到reward路途中不可或缺的一环
        # Method 1:
        # self.eligibility.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # 随着时间衰减eligibility trace的值，离获得reward越远的步，它的"不可或缺性"越小
        self.eligibility_trace *= self.gamma * self.lambda_