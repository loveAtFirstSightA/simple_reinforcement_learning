# 递减的贪婪算法 - 探索的欲望逐渐降低

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

# 贪婪算法
def choose_one():
    # 原始贪婪算法
    # # 有小概率随机选择一根拉杆 
    # if random.random() < 0.01: # 探索
    #     return random.randint(0, 9)
    
    # 递减贪婪算法 积累的经验越多，探索的欲望越低 因为我已经把所有按钮中奖的概率都已经摸的的差不多了
    # 求出现在已经玩了多少次
    played_count = sum([len(i) for i in rewards])
    # 随机选择的概率逐渐下降
    if random.random() < 1 / played_count: # 玩的次数多了 分母大了 随机探索的欲望降低了
        return random.randint(0, 9)
    
    # 计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards] # 利用
    # 选择期望奖励最大的拉杆
    return np.argmax(rewards_mean)

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

