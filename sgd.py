# -*- coding=utf-8 -*-

# ax+by+c
import math

lr = 0.1
x0 = 0.001
y0 = 0.001
last = 0.0001


def sgd(a, b, c, lr):
    temX0 = x0
    temY0 = y0
    temX1 = temX0 - lr * (a + b * temY0 + c)
    temY1 = temY0 - lr * (b + a * temX0 + c)
    while math.fabs(a * (temX1 ** 2 - temX0 ** 2) + b * (temY1 ** 2 - temY0 ** 2)) >= last:
        temX0 = temX1
        temY0 = temY1
        temX1 = temX0 - lr * (2 * a * temX0 + b * temY0 * temY0 + c)
        temY1 = temY0 - lr * (2 * b * temY0 + a * temX0 * temX0 + c)
    print(temX0, temY0, sep=",")


sgd(2, 3, 1, lr)
