# Reinforcement learning algorithm

# 汤普森采样算法

import random
import numpy as np
import sys
from loguru import logger
logger.remove()
logger.add(
    sys.stdout,
    format="[{time:YYYY-MM-DD HH:mm:ss.SSS}] [{level}] {message}",
    level="INFO"
)

# 假设十台老虎机，每个老虎机中奖的概率是 0-1 之间的均匀分布
probs = np.random.uniform(size=10)

# 记录每台老虎机的返回值
# 初始化每个老虎机的中间期望为1 不能为0 为零的话可能存在某些老虎机永远都不会被测试到
rewards = [[1] for _ in range(10)]
logger.info("probs: {}", probs)
logger.info("rewards: {}", rewards)


def choose_one():
    # 求出每个老虎机出1的次数 +1
    count_1 = [sum(i) + 1 for i in rewards]
    
    # 求出每个老虎机出0的次数 +1
    count_0 = [sum(1 - np.array(i)) + 1 for i in rewards]
    
    # 按照beta分布计算奖励分布，这可以认为是每台老虎机中奖的概率
    # 根据beta分布的测试来看，每台老虎机被玩的次数越多，它的分布会越来越稳定，也就是说越来越趋向于利用
    beta = np.random.beta(count_1, count_0)
    
    return beta.argmax()

# 函数功能 选择有一台老虎机玩并得到奖励结果 中奖reward == 1 不中奖 reward == 0
# 并把每次玩的结果保存下载
def try_and_play():
    i = choose_one()
    # 玩老虎机， 得到结果
    reward = 0
    if random.random() < probs[i]:
        reward = 1
    # 记录玩的结果
    rewards[i].append(reward)

def get_result():
    # 玩N次
    for _ in range(5000):
        try_and_play()
    # 期望的最好结果
    target = probs.max() * 5000
    # 实际玩出的结果
    result = sum([sum(i) for i in rewards])
    logger.info("target: {}, result: {}, cost: {}", target, result, target - result)
    return target, result

if __name__ == '__main__':
    get_result()

