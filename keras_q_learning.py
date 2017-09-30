import numpy as np
import pandas as pd
import time

np.random.seed(2017)

N_STATES = 6  # 1维世界的宽度
ACTIONS = ['left', 'right']  # 探索者的可用动作
EPSILON = 0.9  # 贪婪度 greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间
