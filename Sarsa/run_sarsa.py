from maze_env import Maze
from Sarsa import SarsaTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # Sarsa根据state观测选择行为
        # Sarsa的action是在每一次探索之前选的
        action = RL.choose_action(str(observation))

        while True:
            #fresh env
            env.render()

            # 在环境中采取行为，获得下一个state_(也就是observation_), reward, 和是否中止
            observation_, reward, done = env.step(action)

            # 根据下一个state(observation_)选取下一个action_
            # Sarsa中下一个action就必然会采取
            action_ = RL.choose_action(str(observation_))

            # 从(s, a, r, s_, a_)中学习，跟新Q_table参数
            RL.learn(str(observation), action, reward,
                     str(observation_), action_)

            # 将下一个当成下一步的state(observation) 和 action
            observation = observation_
            action = action_

            # 终止时跳出循环
            if done:
                break
        
    # 所有episodes循环结束
    print('game over')
    env.destroy()

if(__name__ == '__main__'):
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
