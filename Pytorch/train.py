import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import *
import time
import os
from tqdm import tqdm
import datetime
import matplotlib
matplotlib.use('TkAgg')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 数据转换和加载数据集
data_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(20),  # 随机旋转
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # 随机仿射变换
    transforms.ToTensor(),
])

data_val_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])
#训练数据集
batch_size = 256
data_folder = './data/train'
train_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#加载验证数据集
data_folder = './data/val'
val_dataset = datasets.ImageFolder(root=data_folder, transform=data_val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#  创建一个VGG19模型，传入模型名称，并设置类别数为7
model_name = "VGG19"
model = VGG(num_classes=7, vgg_name=model_name)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
#定义了训练的轮数，即模型会遍历整个训练数据集100次
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
start_time = time.time()

# 获取当前时间
current_time = datetime.datetime.now()
file_train_name = os.path.join("logs", current_time.strftime(f"{model_name}_training_results.txt"))  # 记录数据
file_val_name = os.path.join("logs", current_time.strftime(f"{model_name}_validation_results.txt"))

# 这个变量将用于存储到目前为止观察到的最高训练准确率
epoch_acc_best = 0
#循环遍历预定的训练轮数
for epoch in range(num_epochs):
    # 训练
    model.train()
    #用于累加当前轮次的总损失和正确预测的数量
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_predictions / len(train_dataset)

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    print(f'Epoch Train {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

    # 打开文件，如果不存在则创建新文件
    with open(file_train_name, "a") as file:
        # 将数据以指定格式写入文件
        file.write(f'{epoch + 1}/{num_epochs} {epoch_loss:.4f} {epoch_acc:.4f}\n')

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct_predictions = 0
        #遍历训练数据集的每个 batch，进行前向传播和反向传播
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            #计算当前训练轮次的平均损失函数值和平均准确率。
        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = correct_predictions / len(val_dataset)

        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        print(f'Epoch Val {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # 打开文件，如果不存在则创建新文件
        with open(file_val_name, "a") as file:
            # 将数据以指定格式写入文件
            file.write(f'{epoch + 1}/{num_epochs} {epoch_loss:.4f} {epoch_acc:.4f}\n')

        if epoch_acc > epoch_acc_best:
            epoch_acc_best = epoch_acc
            model_state_dict_best = model.state_dict()
            print(f"更新了模型，{epoch_acc_best:.4f}")

print(f"训练时间：{time.time() - start_time}")

# 保存模型
torch.save(model_state_dict_best, f'./weights/{model_name}.pth')

# 绘制acc和loss曲线
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"./images/{model_name}.png", dpi=300)
plt.show()
