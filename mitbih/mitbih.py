import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from imblearn.over_sampling import SMOTE

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 加载数据集
train_data = pd.read_csv('mitbih_train.csv', header=None)
test_data = pd.read_csv('mitbih_test.csv', header=None)

# 假设标签在最后一列，特征在前面
X_train = train_data.iloc[:, :-1].values  # 所有行，除去最后一列作为特征
y_train = train_data.iloc[:, -1].values  # 最后一列为标签

X_test = test_data.iloc[:, :-1].values  # 所有行，除去最后一列作为特征
y_test = test_data.iloc[:, -1].values  # 最后一列为标签

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # 对测试集进行相同的标准化

# 转换为Tensor
x_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)  # 添加一个维度，LSTM需要三维输入
x_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 2. 计算类别权重
unique_labels = np.unique(y_train.numpy())
print(f"Unique labels in y_train: {unique_labels}")

# 计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 打印类别权重
print(f"Class weights: {class_weights}")

# 3. 使用SMOTE过采样来平衡数据集
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 转换为Tensor
x_train_resampled = torch.tensor(X_train_resampled, dtype=torch.float32).unsqueeze(2)
y_train_resampled = torch.tensor(y_train_resampled, dtype=torch.long)


# 4. 定义LSTM模型
class ECG_LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, output_size=5):  # hidden_layer_size=32
        super(ECG_LSTM_Model, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.2)  # Dropout层

        # 全连接层
        self.fc = nn.Linear(hidden_layer_size, output_size)  # output_size=5

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # LSTM的最后一步输出经过Dropout
        output = self.fc(lstm_out)
        return output


# 5. 定义Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, num_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# 6. 初始化模型、损失函数和优化器
model = ECG_LSTM_Model(input_size=1, hidden_layer_size=32, output_size=5)  # hidden_layer_size=32
model = model.to(device)  # 将模型移到GPU或CPU

# 使用交叉熵损失和Adam优化器，传入class_weights
criterion = FocalLoss(gamma=2, alpha=0.25, num_classes=5)  # 使用Focal Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 使用DataLoader进行批处理
batch_size = 64  # 尝试减小批量大小

# 创建TensorDataset和DataLoader
train_dataset_resampled = TensorDataset(x_train_resampled, y_train_resampled)
test_dataset = TensorDataset(x_test, y_test)

train_loader_resampled = DataLoader(train_dataset_resampled, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 混合精度训练的Scaler
scaler = GradScaler()

# 8. 训练模型
num_epochs = 30
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# 使用tqdm显示训练进度
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader_resampled:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU或CPU

        optimizer.zero_grad()  # 每次迭代前清空梯度

        # 混合精度训练
        with autocast():  # 使用自动混合精度
            # 前向传播
            outputs_train = model(inputs)
            loss_train = criterion(outputs_train, labels)

        # 反向传播
        scaler.scale(loss_train).backward()  # 使用scaler处理反向传播
        scaler.step(optimizer)  # 使用scaler进行优化步
        scaler.update()  # 更新scaler状态

        # 计算训练集准确率
        _, predicted_train = torch.max(outputs_train, 1)
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)

        running_loss += loss_train.item()

    accuracy_train = correct_train / total_train
    avg_train_loss = running_loss / len(train_loader_resampled)

    # 清空缓存
    torch.cuda.empty_cache()

    # 计算测试集损失和准确率
    model.eval()  # 设置模型为评估模式
    correct_test = 0
    total_test = 0
    running_loss_test = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU或CPU

            with autocast():  # 使用自动混合精度
                outputs_test = model(inputs)
                loss_test = criterion(outputs_test, labels)

            running_loss_test += loss_test.item()

            _, predicted_test = torch.max(outputs_test, 1)
            correct_test += (predicted_test == labels).sum().item()
            total_test += labels.size(0)

    accuracy_test = correct_test / total_test
    avg_test_loss = running_loss_test / len(test_loader)

    # 保存损失和准确率
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(accuracy_train)
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(accuracy_test)

    # 使用tqdm显示进度条
    tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], "
               f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, "
               f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy_test:.4f}")

# 9. 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    outputs_test = model(x_test)
    _, predicted_test = torch.max(outputs_test, 1)
    accuracy_test = (predicted_test == y_test).sum().item() / len(y_test)
    print(f"Final Test Accuracy: {accuracy_test:.4f}")

    # 输出分类报告
    print(classification_report(y_test.cpu(), predicted_test.cpu()))  # 转回CPU进行报告生成

# 10. 绘制训练过程中的损失和准确率
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
