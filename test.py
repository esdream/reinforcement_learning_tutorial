# -*- coding: utf-8 -*-
"""Test
测试某些方法或代码
"""
import numpy as np
import pandas as pd

"""
Q-learning
"""
test1 = pd.DataFrame([[1, ""], [0, 4]])

print(test1)

# all(axis=None) 根据axis返回axis方向上是否所有的元素都为True(0，""，等都是False)。默认axis是按列计算
print(test1.all())

# np.random.choice()随机选择元素
print(np.random.choice(['left', 'right']))


"""
Sarsa & Sarsa lambda
"""

a = pd.DataFrame([[1, 2], [3, 4]], columns=['name', 'age'], index=['Alias', 'Bob'])
# 在series中，index相当于dataframe中的columns, name相当于dataframe中的index
print(a.append(pd.Series([0] * 2, index=a.columns, name='right')))