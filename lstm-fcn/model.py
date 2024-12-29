import torch
import numpy as np
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from modules import LSTM_FCN


class Model():
    def __init__(self, x, y, units, dropout, filters, kernel_sizes,normalize=True):

        '''
        参数：
        __________________________________
        x: np.array.
            时间序列数据，数组形状为(samples, length)，其中samples是时间序列的样本数量，length是每个时间序列的长度。

        y: np.array.
            类别标签，数组形状为(samples,)，其中samples是时间序列的样本数量。

        units: list of int.
            列表长度对应LSTM块的数量，每个列表项表示该LSTM层的单元数。

        dropout: float.
            每个LSTM块后应用的dropout比例。

        filters: list of int.
            列表长度对应卷积块的数量，每个列表项表示该卷积层的滤波器（或通道）数。

        kernel_sizes: list of int.
            列表长度对应卷积块的数量，每个列表项表示该卷积层的卷积核大小。
        '''

        # 检查是否有可用的GPU。
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalize=normalize
        # 对数据进行洗牌。
        x, y = shuffle(x, y)

        # 对数据进行归一化处理。
        self.x_min = np.nanmin(x, axis=0, keepdims=True)
        self.x_max = np.nanmax(x, axis=0, keepdims=True)
        x = (x - self.x_min) / (self.x_max - self.x_min)

        # 计算类别的权重，用于处理类别不平衡问题。
        self.weight = compute_class_weight(class_weight="balanced", classes=np.sort(np.unique(y)), y=y)
        print(len(np.unique(y)))
        # 构建模型。
        model = LSTM_FCN(
            input_length=x.shape[1],
            units=units,
            dropout=dropout,
            filters=filters,
            kernel_sizes=kernel_sizes,
            num_classes=len(np.unique(y))
        )

        # 保存数据。
        self.x = torch.from_numpy(x).to(self.device).float()
        self.y = torch.from_numpy(y).to(self.device).long()

        # 保存模型。
        self.model = model.to(self.device)

    def fit(self, learning_rate, batch_size, epochs, verbose=True):

        '''
        训练模型。

        参数：
        __________________________________
        learning_rate: float.
            学习率。

        batch_size: int.
            批量大小。

        epochs: int.
            训练的轮数。

        verbose: bool.
            如果为True，控制台将输出训练的历史记录；如果为False，则不输出。
        '''

        # 生成训练数据的批次。
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True
        )

        # 定义优化器。
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 定义损失函数，使用加权的交叉熵损失。
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(self.weight).float().to(self.device))

        # 训练模型。
        self.model.train(True)
        print(f'正在 {self.device} 上进行训练。')
        for epoch in range(epochs):
            for features, targets in dataset:
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                accuracy = (torch.argmax(torch.nn.functional.softmax(outputs, dim=-1),
                                         dim=-1) == targets).float().sum() / targets.shape[0]
            if verbose:
                print('第{}轮，损失：{:,.6f}，准确率：{:.6f}'.format(1 + epoch, loss, accuracy))
        self.model.train(False)

    def predict(self, x):

        '''
        预测类别标签。

        参数：
        __________________________________
        x: np.array.
            时间序列数据，数组形状为(samples, length)，其中samples是时间序列的样本数量，length是每个时间序列的长度。

        返回：
        __________________________________
        y: np.array.
            预测的标签，数组形状为(samples,)，其中samples是时间序列的样本数量。
        '''

        # 对输入数据进行归一化处理。
        if(self.normalize):
            x = (x - self.x_min) / (self.x_max - self.x_min)

        # 获取预测的概率。
        p = torch.nn.functional.softmax(self.model(torch.from_numpy(x).to(self.device).float()), dim=-1)

        # 获取预测的标签。
        y = np.argmax(p.detach().cpu().numpy(), axis=-1)

        return y
