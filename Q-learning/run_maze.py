from maze_env import Maze
from RL_brain import QLearningTable

def update():
    # 学习100回合
    for episode in range(100):
        # 初始化state的观测值
        observation = env.reset()
        
        while True:
            # 更新可视化环境
            env.render()

            # RL大脑根据state的观测值挑选action
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if(__name__ == '__main__'):
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
