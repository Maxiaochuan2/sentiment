import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import VGG
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')

# 数据转换和加载测试数据集
data_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

data_folder = './data/test'
test_dataset = datasets.ImageFolder(root=data_folder, transform=data_transform)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_name = "VGG19"
model = VGG(num_classes=7, vgg_name=model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型权重
model_weights_path = f'./weights/{model_name}.pth'
model.load_state_dict(torch.load(model_weights_path,map_location=device))
model.eval()

# 在测试集上进行预测
model.to(device)

all_labels = []
all_predictions = []
all_probs = []
criterion = nn.CrossEntropyLoss()  # 定义损失函数
total_loss = 0.0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Validation', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(outputs.cpu().numpy())

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 绘制混淆矩阵热力图
plt.figure(figsize=(19, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=35)
os.makedirs("./images", exist_ok=True)
plt.savefig(f"./images/Confusion_{model_name}.png")
plt.close()

# 输出分类报告
class_report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes, digits=4)
print("Classification Report:\n", class_report)

# 计算精度、精确率、召回率、F1分数
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 输出总损失值
print(f"Total Loss: {total_loss:.4f}")

# 将分类报告写入txt文件
with open(os.path.join("images", model_name + ".txt"), 'w') as file:
    file.write("Classification Report:\n")
    file.write(class_report)
    file.write(f"\nAccuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")
    file.write(f"Total Loss: {total_loss:.4f}\n")

# 保存错误分类的部分数据
errors_dir = './images/errors'
os.makedirs(errors_dir, exist_ok=True)


# 转换张量为图像的函数
def tensor_to_image(tensor):
    if tensor.dim() == 3 and tensor.size(0) in [1, 3]:  # 确保是三维张量并且通道数是1或3
        # 取消归一化（如果你之前对数据进行了归一化处理）
        tensor = tensor * 255.0
        tensor = tensor.byte()
        np_array = tensor.cpu().numpy()

        if tensor.size(0) == 1:  # 单通道图像
            np_array = np_array[0, :, :]  # [H, W]
            return Image.fromarray(np_array, mode='L')
        elif tensor.size(0) == 3:  # RGB图像
            np_array = np_array.transpose(1, 2, 0)  # 转换为 [H, W, C] 格式
            return Image.fromarray(np_array, mode='RGB')
    else:
        raise ValueError("Unexpected tensor shape: {}".format(tensor.size()))


# 保存前10个错误分类样本
error_count = 0
for i, (input, label, pred) in enumerate(zip(test_dataset, all_labels, all_predictions)):
    if label != pred and error_count < 10:
        img_tensor, true_label = input

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor  # 直接使用张量
        elif isinstance(img_tensor, (tuple, list)):
            img_tensor = img_tensor[0]  # 从输入中提取图像张量部分

        # 转换为 PIL 图像
        img = tensor_to_image(img_tensor)

        img_path = os.path.join(errors_dir, f"error_{i}_true_{label}_pred_{pred}.png")
        img.save(img_path)
        error_count += 1
