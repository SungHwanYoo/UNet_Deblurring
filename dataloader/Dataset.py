import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import glob
import os
from torchvision import transforms

class Cat_Dog_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 고양이와 강아지 이미지 경로 리스트 생성
        self.cat_images = glob.glob(os.path.join(data_dir, 'cats', '*.jpg'))
        self.dog_images = glob.glob(os.path.join(data_dir, 'dogs', '*.jpg'))
        
        # 전체 이미지 경로 리스트
        self.image_paths = self.cat_images + self.dog_images
        
        # 기본 transform 정의
        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),  # 이미지 크기 통일
            transforms.ToTensor(),          # PIL Image를 텐서로 변환 (0-1 범위로 정규화)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        target = Image.open(image_path).convert('RGB')
        
        # 이미지 블러 처리
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # PIL Image를 텐서로 변환
        image = self.to_tensor(image)       # shape: [3, 256, 256]
        target = self.to_tensor(target)     # shape: [3, 256, 256]
        
        # 추가적인 transform이 있다면 적용
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target