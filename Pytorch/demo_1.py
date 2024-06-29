"""
替换名字
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
FER2013 Dataset
"""

import os

# 定义数字到英文字母的映射关系
mapping = {
    '0': 'Angry',
    '1': 'Disgust',
    '2': 'Fear',
    '3': 'Happy',
    '4': 'Sad',
    '5': 'Surprise',
    '6': 'Neutral'
}

# 获取当前文件夹路径
folder_path = './data/test'

# 获取文件夹列表
folders = next(os.walk(folder_path))[1]

# 遍历文件夹并重命名
for folder in folders:
    # 检查是否为数字名称
    if folder.isdigit():
        # 获取对应的英文字母名称
        new_name = mapping[folder]
        # 构建新的路径
        old_path = os.path.join(folder_path, folder)
        new_path = os.path.join(folder_path, new_name)
        # 重命名文件夹
        os.rename(old_path, new_path)
        print(f"Renamed {folder} to {new_name}")
