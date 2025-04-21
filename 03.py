# UCB 上置信界算法

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


# 按照强化学习的定义 - 此处应该是动作函数 action function
# 随机选择概率递减的贪婪算法
def choose_one():
    # 求出每个老虎机各玩了多少次
    played_count = [len(i) for i in rewards]
    played_count = np.array(played_count)
    # logger.info("played_count: {}", played_count)
    
    # 求出上置信界
    # 分子是总共玩了多少次，取根号后让它的增长速度变慢
    # 分母是每台老虎机玩的次数，乘以2让它的增长速度变快
    # 随着玩的次数增加，分母会很快超过分子的增长速度，导致分数越来越小
    # 具体到每一台老虎机，则是玩的越多，分数越小，也就是UCB的加权越小
    # 所以UCB衡量了每一台老虎机的不确定性，不确定性越大，探索的价值越大
    fenzi = played_count.sum() ** 0.5
    fenmu = played_count * 2
    ucb = fenzi / fenmu # 这里ucb不是一个scalar，而是一个向量
    
    # ucb本身取根号
    # 大于1的数字会被缩小，小于1的数字会被放大，这样保持ucb恒定在一定的数值范围内
    ucb = ucb ** 0.5
    
    # 计算每个老虎机的奖励平均
    rewards_mean = [np.mean(i) for i in rewards]
    rewards_mean = np.array(rewards_mean)
    
    ucb += rewards_mean
    
    return ucb.argmax()

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

