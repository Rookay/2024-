import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import Model
from plots import plot

# 生成数据
N = 600  # 时间序列的数量
L = 100  # 每个时间序列的长度
x = np.zeros((N, L))  # 初始化数据矩阵
t = np.linspace(0, 1, L)  # 时间序列的时间步
c = np.cos(2 * np.pi * (10 * t - 0.5))  # 生成cos波
s = np.sin(2 * np.pi * (20 * t - 0.5))  # 生成sin波
x[:N // 3] = 10 + 10 * c + 5 * np.random.normal(size=(N // 3, L))  # 第一类数据：cos波 + 噪声
x[N // 3: 2 * N // 3] = 10 + 10 * s + 5 * np.random.normal(size=(N // 3, L))  # 第二类数据：sin波 + 噪声
x[2 * N // 3:] = 10 + 10 * c + 10 * s + 5 * np.random.normal(size=(N // 3, L))  # 第三类数据：cos波 + sin波 + 噪声
y = np.concatenate([0 * np.ones(N // 3), 1 * np.ones(N // 3), 2 * np.ones(N // 3)])  # 标签：三类（0、1、2）

# 数据划分：将数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

# 定义模型
model = Model(
    x=x_train,  # 训练数据
    y=y_train,  # 训练标签
    units=[5, 5],  # LSTM层的单元数
    filters=[4, 4],  # 卷积层的滤波器数量
    kernel_sizes=[3, 3],  # 卷积核大小
    dropout=0.2,  # Dropout比率
)

train_accuracies = []
test_accuracies = []

# 训练模型并记录准确率
epochs = 100  # 训练的总轮数
for epoch in range(epochs):
    model.fit(
        learning_rate=0.0005,
        batch_size=32,
        epochs=1,
        verbose=False
    )

    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, yhat_train)
    test_accuracy = accuracy_score(y_test, yhat_test)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}/{epochs} - 训练集准确率: {train_accuracy:.4f}, 测试集准确率: {test_accuracy:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_accuracies, label='Training Accuracy', color='blue', linestyle='-', marker='o')
plt.plot(range(epochs), test_accuracies, label='Testing Accuracy', color='red', linestyle='-', marker='x')
plt.title('Accuracy Change During Training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
plt.show()
fig = plot(x, y)
fig.show()
