# -*- coding=utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# 定义数据部分
x = np.arange(0., 10, 0.2)
y = np.arange(0., 10, 0.2)
y1 = np.cos(x)
y2 = np.sin(x)
y3 = np.sqrt(x)
h = 2 * np.sqrt(x) + 3 * np.sqrt(y) + 5

# 绘制 3 条函数曲线
# plt.plot(x, y1, color='blue', linewidth=1.5, linestyle='-', marker='.', label=r'$y = cos{x}$')
# plt.plot(x, y2, color='green', linewidth=1.5, linestyle='-', marker='*', label=r'$y = sin{x}$')
# plt.plot(x, y3, color='m', linewidth=1.5, linestyle='-', marker='x', label=r'$y = \sqrt{x}$')
plt.plot(x, y, h, color='m', linewidth=1.5, linestyle='-', marker='x', label=r'$y = \ax**2+by**2+c$')
# # 坐标轴上移
# ax = plt.subplot(111)
# ax.spines['right'].set_color('none')     # 去掉右边的边框线
# ax.spines['top'].set_color('none')       # 去掉上边的边框线
# # 移动下边边框线，相当于移动 X 轴
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# # 移动左边边框线，相当于移动 y 轴
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))

plt.show()
