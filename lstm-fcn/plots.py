import numpy as np
import matplotlib.pyplot as plt

def plot(x, y):
    '''
    绘制每个类别的时间序列图。

    参数:
    __________________________________
    x: np.array.
        时间序列数据，形状为 (样本数, 长度)，其中：
        - 样本数是时间序列的数量，
        - 长度是时间序列的长度。

    y: np.array.
        预测的标签，形状为 (样本数,)，表示每个时间序列的类别标签。

    返回:
    __________________________________
    fig: plt.Figure.
        时间序列的折线图，每个类别一个子图。
    '''

    c = np.unique(y).astype(int)  # 获取类别标签

    # 创建图形和子图
    fig, axes = plt.subplots(len(c), 1, figsize=(10, 5 * len(c)))

    # 确保 axes 是数组形式，方便后续的索引操作
    if len(c) == 1:
        axes = [axes]

    for i in range(len(c)):
        # 提取当前类别的时间序列数据
        x_ = x[y == c[i], :]

        # 绘制每个时间序列
        for j in range(x_.shape[0]):
            axes[i].plot(x_[j, :], color='lightgray', linewidth=0.5)

        # 设置类别平均线
        avg_line = np.mean(x_, axis=0)
        axes[i].plot(avg_line, color='red', linewidth=2, label=f'Class {i+1} Average')

        # 设置图表的标题和标签
        axes[i].set_title(f'Class {i+1}', fontsize=14)
        axes[i].set_xlabel('Time', fontsize=12)
        axes[i].set_ylabel('Value', fontsize=12)

        # 显示图例
        axes[i].legend()

    # 自动调整布局
    plt.tight_layout()

    return fig
