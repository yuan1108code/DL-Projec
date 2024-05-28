# custom_dataset.py
import torch
import os
import pandas as pd
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_df, root_dir, transform=None):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            image = Image.open(img_name)
            
            # 检查图像通道数，如果是单通道则转换为三通道
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            label = self.label_mapping[self.annotations.iloc[idx, 1]]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(e)
            print("Error loading image:", img_name, "idx", idx)
            return None, None  # 返回 None 值，表示加载失败
