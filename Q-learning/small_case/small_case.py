# -*- coding: utf-8 -*-
"""Small Case
Q-learning的小例子，与目录下其他三个.py文件没有关系
"""

import numpy as np
import pandas as pd
import time

N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间

# Q表
# Q表中的index是所有对应的state（探索者位置），columns是对应的action（探索者行为）
# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # q_table全部初始化为0
        columns=actions
    )
    return table

# 定义动作
# 在某个state地点，选择行为
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :] # 选出state的所有action值, iloc是根据index位置选择行或列
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0): # 非贪婪或者这个state还没有探索过
        action_name = np.random.choice(ACTIONS)  # np.random.choice()从一个Int数字或1维array里随机选取内容
    else:
        action_name = state_actions.argmax() # argmax(axis=0)返回axis上的最大值, 0 for index, 1 for columns
    return action_name

# 环境反馈S_, R
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if(A == 'right'):
        if(S == N_STATES - 2):
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if(S == 0):
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R

# 环境更新
def update_env(S, episode, step_counter):
    # Thie is how environment be udpated
    env_list = ['-']*(N_STATES-1) + ['T']
    if(S == 'terminal'):
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        # end 表示不换行本行更新
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                      ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# 强化学习主循环
def rl():
    q_table = build_q_table(N_STATES, ACTIONS) # 初始化q table
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0 # 回合初始位置
        is_terminated = False # 是否回合结束
        update_env(S, episode, step_counter)

        while not is_terminated:
            A = choose_action(S, q_table) # 选择行为
            S_, R = get_env_feedback(S, A) # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A] # 估算的(状态-行为)值
            if(S_ != 'terminal'):
                q_target = R + GAMMA * q_table.iloc[S_, :].max() # 实际的(状态-行为)值(回合没有结束)
            else:
                q_target = R # 实际的(状态-行为)值(回合结束)
                is_terminated = True # terminate this episode
            
            q_table.loc[S, A] += ALPHA * (q_target - q_predict) # q_table更新
            S = S_ # 探索者移动至下一个state

            update_env(S, episode, step_counter+1) # 环境更新
            
            step_counter += 1
    
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)