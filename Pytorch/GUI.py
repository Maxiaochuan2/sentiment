import sys
import os
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel, QMessageBox, \
    QTextEdit, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
from model import VGG
from sklearn.metrics import classification_report
from torchvision import datasets

class ImageClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.model_name = "VGG19"
        self.model = VGG(num_classes=7, vgg_name=self.model_name)
        self.model_weights_path = f'./weights/{self.model_name}.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])
        self.test_dataset = datasets.ImageFolder(root='./data/test', transform=self.transform)
        self.mapping = {
            '0': '生气',
            '1': '厌恶',
            '2': '恐惧',
            '3': '高兴',
            '4': '悲伤',
            '5': '惊讶',
            '6': '平静'
        }

    def initUI(self):
        self.setWindowTitle('情绪识别')
        self.setGeometry(100, 100, 800, 600)


        main_layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel('情绪识别', self)
        title_label.setStyleSheet('font-size: 24px; font-weight: bold; padding: 10px;')
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        # 添加系统使用说明
        description_label = QLabel(
            '系统说明：'
            '请点击“打开图片”选择一张图片，然后点击“分类图片”进行情绪识别。识别结果将显示在下方。',
            self
        )
        description_label.setStyleSheet('font-size: 14px; padding: 10px;')
        main_layout.addWidget(description_label, alignment=Qt.AlignCenter)
        # 创建按钮布局
        button_layout = QHBoxLayout()
        self.btnOpen = QPushButton('打开图片', self)
        self.btnOpen.clicked.connect(self.openImage)
        button_layout.addWidget(self.btnOpen)
        self.btnClassify = QPushButton('分类图片', self)
        self.btnClassify.clicked.connect(self.classifyImage)
        button_layout.addWidget(self.btnClassify)
        main_layout.addLayout(button_layout)
        # 在initUI方法中添加背景色和按钮颜色
        self.setStyleSheet("background-color: #f2f2f2;")
        # 修改按钮样式
        self.btnOpen.setStyleSheet(
            "background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px;")
        self.btnClassify.setStyleSheet(
            "background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px;")

        # 在openImage方法中添加选项，以显示文件选择对话框的背景色
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog  # 使用自定义样式

        # 创建图片和结果布局
        image_layout = QHBoxLayout()
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.imageLabel)
        main_layout.addLayout(image_layout)



        text_layout = QVBoxLayout()
        self.textOutput = QTextEdit(self)
        self.textOutput.setReadOnly(True)
        text_layout.addWidget(self.textOutput)
        main_layout.addLayout(text_layout)

        self.setLayout(main_layout)
        self.image_path = None
        self.classifier = None

    def openImage(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if filePath:
            self.image_path = filePath
            pixmap = QPixmap(filePath)
            self.imageLabel.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def classifyImage(self):
        if not self.image_path:
            QMessageBox.warning(self, "错误", "请先打开一张图片!")
            return

        image = Image.open(self.image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = self.mapping[str(predicted.item())]
        self.textOutput.setStyleSheet("background-color: #ffffcc;")
        self.textOutput.append(f"预测结果: {predicted_label}")
        self.saveImageWithLabel(self.image_path, predicted_label)


    def saveImageWithLabel(self, image_path, label):
        base_path, filename = os.path.split(image_path)
        new_filename = f"{os.path.splitext(filename)[0]}_classified_{label}.png"
        output_path = os.path.join(base_path, new_filename)

        image = Image.open(image_path).convert('RGB')
        image.save(output_path)

        self.textOutput.append(f"图像保存为:  {output_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageClassifierGUI()
    ex.show()
    sys.exit(app.exec_())