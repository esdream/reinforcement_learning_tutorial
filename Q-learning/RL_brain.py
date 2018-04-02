"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    # 初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate # 学习率
        self.gamma = reward_decay # 奖励衰减
        self.epsilon = e_greedy # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 选择行为
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # np.random.permutation()返回一个洗牌后的state_action.index副本
            # 由于idxmax()在选择最大值时永远都是选择第一个index，因此需要在这里将state_action的index洗牌，这样保证所有action在最大值相等时会有均等的概率选中
            state_action = state_action.reindex(np.random.permutation(
                state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 学习更新参数
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    # 检测state是否存在
    # 由于不知道q_table的大小（即起点到终点有多少state），因此每次都要检查state是否存在，如果为空则创建并添加到q_table的末尾
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
