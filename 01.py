# 贪婪算法
# 参考链接： https://www.bilibili.com/video/BV1Ge4y1i7L6?spm_id_from=333.788.videopod.episodes&vd_source=c8ccadb8c866c562af98f5dbcfef3b44&p=2

"""
使用场景：
    无状态问题
示例：
    老虎机场景问题，假设有四个中奖按钮， 每个按钮的中奖概率位置，在优先尝试次数内求最大回报？
思路：
    将尝试次数分成两部分，一部分是探索，另外一部分叫作利用
    探索的部分就是我不知道这四个按钮的中奖概率情况，需要尝试按这四个按钮，然后估算出每个按钮的中奖概率的大致情况
    利用就是我已经估算出每个按钮的中奖概率情况，然后我就每次按这个按钮
    探索和应用都要占据有限次数，所以探索和应用的比例是这个问题的关键
贪婪算法：
    大概率选择目前中奖概率最高的，小概率随机探索
"""

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
    # 有小概率随机选择一根拉杆 
    if random.random() < 0.01: # 探索
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

