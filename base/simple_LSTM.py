# -*- coding: UTF-8 -*-
# Project ：demo 
# File    ：simple_LSTM.py
# IDE     ：PyCharm 
# Author  ：刘景涛
# Date    ：2024/12/16 16:29
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 定义一个序列模型
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # 定义两个LSTMCell层和一个全连接层
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    # 定义前向传播
    def forward(self, input, future = 0):
        outputs = []
        # 初始化LSTMCell的隐藏状态和细胞状态
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        # 遍历输入序列
        for input_t in input.split(1, dim=1):
            # 更新LSTMCell的隐藏状态和细胞状态
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # 通过全连接层得到输出
            output = self.linear(h_t2)
            outputs += [output]
        # 如果需要预测未来值
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        # 将输出序列拼接起来
        outputs = torch.cat(outputs, dim=1)
        return outputs
