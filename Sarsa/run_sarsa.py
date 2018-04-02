from maze_env import Maze
from Sarsa import SarsaTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()