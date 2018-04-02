# -*- coding: utf-8 -*-
"""
以指定概率选取元素
"""
import random

def random_pick(some_list, probabilities):
    """按概率随机返回元素
    args:
        som_list: 元素
        probabilities: 每个元素指定的概率 
    """
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    # zip函数将两个list分别拆解，按元素成对的方式返回
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if(x < cumulative_probability):
            break
    return item

def test_random(times):
    """测试
    """
    # 元素
    items = [1, 2, 3, 4]
    # 元素对应概率
    probs = [0.4, 0.1, 0.3, 0.2]

    res = dict(zip(items, [0] * 4))
    for _ in range(times):
        item = random_pick(items, probs)
        res[item] += 1

    for item, item_times in res.items():
        res[item] = float(item_times) / times

    return res

if(__name__ == '__main__'):
    res = test_random(100000)
    print(res)
    # for _ in range(20):
    #     item = random_pick([1, 2], [0.1, 0.9])
    #     print(item)
