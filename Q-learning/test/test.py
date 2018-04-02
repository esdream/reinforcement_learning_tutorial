# -*- coding: utf-8 -*-
"""Test
测试某些方法或代码
"""
import numpy as np
import pandas as pd

test1 = pd.DataFrame([[1, ""], [0, 4]])

print(test1)

# all(axis=None) 根据axis返回axis方向上是否所有的元素都为True(0，""，等都是False)。默认axis是按列计算
print(test1.all())

print(np.random.choice(['left', 'right']))
