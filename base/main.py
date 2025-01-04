from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

from simple_LSTM import Sequence

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=8, help='steps to run')
opt = parser.parse_args()

def data():
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)

    # 加载数据并构建训练集
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    return input, target, test_input, test_target

if __name__ == '__main__':
    # 构建模型
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # 使用LBFGS作为优化器，因为我们可以将所有数据加载到训练中
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    input, target, test_input, test_target=data()
    # 开始训练
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # 开始预测，不需要跟踪梯度
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # 绘制结果
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.png' % i)
        plt.close()

