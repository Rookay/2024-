import matplotlib.pyplot as plt

# 初始化列表以存储每个指标的值
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# 读取文件并解析数据
with open('log.txt', 'r') as file:
    for line in file:
        # 假设每行数据格式为 "Epoch [X/30], Train Loss: Y, Train Accuracy: Z, Test Loss: A, Test Accuracy: B"
        parts = line.split(',')
        epoch = int(parts[0].split('/')[0].strip('Epoch []'))
        train_loss = float(parts[1].split(': ')[1])
        train_acc = float(parts[2].split(': ')[1])
        test_loss = float(parts[3].split(': ')[1])
        test_acc = float(parts[4].split(': ')[1])

        # 将解析后的值添加到对应的列表中
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

# 绘制Loss图形
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制Accuracy图形
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()