import numpy as np
import random

print("beta分布测试, 输入参数有两个")

print("当数字小的时候，beta分布的概率有很大的随机性")
for _ in range(5):
    print(np.random.beta(1, 1))

print("当数字大的时候， beta分布逐渐稳定")
for _ in range(5):
    print(np.random.beta(1e5, 1e5))

