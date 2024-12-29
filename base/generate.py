# -*- coding: UTF-8 -*-
# Project ：demo 
# File    ：generate.py
# IDE     ：PyCharm 
# Author  ：刘景涛
# Date    ：2024/12/16 16:29
import numpy as np
import torch

# 设置随机种子，以确保结果的可重复性
np.random.seed(2)

# 定义常数 T、L 和 N
T = 20
L = 1000
N = 100

# 创建一个空的 numpy 数组 x，用于存储生成的序列
x = np.empty((N, L), 'int64')

# 为数组 x 赋值，每行都是一个按顺序排列的整数序列，
# 并加入了一个随机偏移量
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)

# 对 x 进行正弦变换，以此生成正弦波数据
data = np.sin(x / 1.0 / T).astype('float64')

# 将生成的正弦波数据保存为一种 PyTorch 可读的格式
torch.save(data, open('traindata.pt', 'wb'))
