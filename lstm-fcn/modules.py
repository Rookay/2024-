import torch
import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')


class LSTM(torch.nn.Module):

    def __init__(self, input_length, units, dropout):

        '''
        参数：
        __________________________________
        input_length: int.
            时间序列的长度。

        units: list of int.
            列表的长度对应LSTM块的数量，列表中的每个元素表示该LSTM层的单元数。

        dropout: float.
            每个LSTM块后应用的dropout比率。
        '''

        super(LSTM, self).__init__()

        # 检查输入的units是否为列表
        if type(units) != list:
            raise ValueError(f'单元数应该以列表形式提供。')

        # 构建模型
        modules = OrderedDict()
        for i in range(len(units)):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=input_length if i == 0 else units[i - 1],
                # 第一层LSTM的输入维度为input_length，后续层的输入维度为上一层的hidden_size
                hidden_size=units[i],  # 当前LSTM层的单元数
                batch_first=True  # 使得输入数据的形状为(batch_size, seq_len, input_size)
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])  # 选择LSTM的输出（即，最后一个时间步的输出）
            modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)  # 在每个LSTM块后添加dropout
        self.model = torch.nn.Sequential(modules)  # 将所有模块组合成一个序列

    def forward(self, x):

        '''
        参数：
        __________________________________
        x: torch.Tensor.
            输入的时间序列，张量形状为(samples, 1, length)，其中samples是时间序列的样本数量，length是每个时间序列的长度。
        '''

        return self.model(x)[:, -1, :]  # 返回最后一个时间步的LSTM输出


class FCN(torch.nn.Module):

    def __init__(self, filters, kernel_sizes):

        '''
        参数：
        __________________________________
        filters: list of int.
            列表的长度对应卷积块的数量，每个元素表示该卷积层的滤波器（或通道）数。

        kernel_sizes: list of int.
            列表的长度对应卷积块的数量，每个元素表示该卷积层的卷积核大小。
        '''

        super(FCN, self).__init__()

        # 检查filters和kernel_sizes的长度是否相同
        if len(filters) == len(kernel_sizes):
            blocks = len(filters)
        else:
            raise ValueError(f'滤波器数量和卷积核大小必须相同。')

        # 构建卷积神经网络（FCN）
        modules = OrderedDict()
        for i in range(blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(
                in_channels=1 if i == 0 else filters[i - 1],  # 第一层卷积的输入通道数为1，后续为前一层的输出通道数
                out_channels=filters[i],  # 当前卷积层的输出通道数
                kernel_size=(kernel_sizes[i],),  # 卷积核的大小
                padding='same'  # 保证输出和输入长度相同
            )
            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(
                num_features=filters[i],  # 归一化的特征数
                eps=0.001,
                momentum=0.99
            )
            modules[f'ReLU_{i}'] = torch.nn.ReLU()  # 每层卷积后添加ReLU激活函数
        self.model = torch.nn.Sequential(modules)  # 将所有模块组合成一个序列

    def forward(self, x):

        '''
        参数：
        __________________________________
        x: torch.Tensor.
            输入的时间序列，张量形状为(samples, 1, length)，其中samples是时间序列的样本数量，length是每个时间序列的长度。
        '''

        return torch.mean(self.model(x), dim=-1)  # 输出FCN的均值特征


class LSTM_FCN(torch.nn.Module):

    def __init__(self, input_length, units, dropout, filters, kernel_sizes, num_classes):
        '''
        参数：
        __________________________________
        input_length: int.
            时间序列的长度。

        units: list of int.
            列表的长度对应LSTM块的数量，每个元素表示该LSTM层的单元数。

        dropout: float.
            每个LSTM块后应用的dropout比率。

        filters: list of int.
            列表的长度对应卷积块的数量，每个元素表示该卷积层的滤波器（或通道）数。

        kernel_sizes: list of int.
            列表的长度对应卷积块的数量，每个元素表示该卷积层的卷积核大小。

        num_classes: int.
            类别的数量。
        '''

        super(LSTM_FCN, self).__init__()

        # 创建FCN模块
        self.fcn = FCN(filters, kernel_sizes)

        # 创建LSTM模块
        self.lstm = LSTM(input_length, units, dropout)

        # 创建全连接层，输出类别的数量
        self.linear = torch.nn.Linear(in_features=filters[-1] + units[-1], out_features=num_classes)

    def forward(self, x):
        '''
        参数：
        __________________________________
        x: torch.Tensor.
            输入的时间序列，张量形状为(samples, length)，其中samples是时间序列的样本数量，length是每个时间序列的长度。

        返回：
        __________________________________
        y: torch.Tensor.
            输出的logits，张量形状为(samples, num_classes)，其中samples是时间序列的样本数量，num_classes是类别的数量。
        '''

        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))  # 重新调整输入的形状，使其符合卷积层的输入要求
        y = torch.concat((self.fcn(x), self.lstm(x)), dim=-1)  # 将FCN和LSTM的输出连接起来
        y = self.linear(y)  # 通过全连接层输出最终的预测结果

        return y


class Lambda(torch.nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)  # 应用传入的lambda函数
