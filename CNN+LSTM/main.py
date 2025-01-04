import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm

# 1. 加载数据集
normal_data = pd.read_csv('data/archive/ptbdb_normal.csv', header=None)
abnormal_data = pd.read_csv('data/archive/ptbdb_abnormal.csv', header=None)

# 标记标签：正常为0，异常为1
normal_data['label'] = 0
abnormal_data['label'] = 1

# 合并正常与异常数据
data = pd.concat([normal_data, abnormal_data], axis=0)

# 特征和标签
X = data.iloc[:, :-1].values  # 所有行，除去最后一列作为特征
y = data.iloc[:, -1].values  # 最后一列为标签

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 计算类权重（处理不平衡数据）
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 转换为Tensor
x_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 添加通道维度，CNN需要三维输入 (批次大小, 通道数, 序列长度)
x_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 2. 定义LSTM + CNN模型
class ECG_LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=2):
        super(ECG_LSTM_CNN_Model, self).__init__()

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM层
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_layer_size, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # CNN提取特征
        cnn_out = self.cnn(x)  # 输出形状: (批次大小, 64, 序列长度 / 4)
        cnn_out = cnn_out.permute(0, 2, 1)  # 调整形状为 (批次大小, 序列长度 / 4, 64) 以适应 LSTM 输入

        # LSTM处理时间序列
        lstm_out, _ = self.lstm(cnn_out)  # 输出形状: (批次大小, 序列长度 / 4, hidden_layer_size)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层分类
        output = self.fc(lstm_out)
        return output


# 3. 初始化模型、损失函数和优化器
model = ECG_LSTM_CNN_Model(input_size=1, hidden_layer_size=64, output_size=2)

# 使用交叉熵损失和Adam优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 20
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# 使用tqdm显示训练进度
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()

    # 前向传播
    outputs_train = model(x_train)
    loss_train = criterion(outputs_train, y_train)

    # 反向传播
    loss_train.backward()
    optimizer.step()

    # 计算训练集准确率
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs_train = model(x_train)
        _, predicted_train = torch.max(outputs_train, 1)
        correct_train = (predicted_train == y_train).sum().item()
        accuracy_train = correct_train / len(y_train)

    # 计算测试集损失和准确率
    with torch.no_grad():
        outputs_test = model(x_test)
        loss_test = criterion(outputs_test, y_test)
        _, predicted_test = torch.max(outputs_test, 1)
        correct_test = (predicted_test == y_test).sum().item()
        accuracy_test = correct_test / len(y_test)

    # 保存损失和准确率
    train_loss_list.append(loss_train.item())
    train_acc_list.append(accuracy_train)
    test_loss_list.append(loss_test.item())
    test_acc_list.append(accuracy_test)

    # 使用tqdm显示进度条
    tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], "
               f"Train Loss: {loss_train.item():.4f}, Train Accuracy: {accuracy_train:.4f}, "
               f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {accuracy_test:.4f}")

# 5. 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    outputs_test = model(x_test)
    _, predicted_test = torch.max(outputs_test, 1)
    accuracy_test = (predicted_test == y_test).sum().item() / len(y_test)
    print(f"Final Test Accuracy: {accuracy_test:.4f}")

    # 输出分类报告
    print(classification_report(y_test, predicted_test))

# 6. 绘制训练过程中的损失和准确率
plt.figure(figsize=(12, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()