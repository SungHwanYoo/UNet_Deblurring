import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataloader.Dataset import Cat_Dog_Dataset

def show_image_pair(dataset, index=0)
    # 데이터셋에서 이미지 쌍 가져오기
    blurred_img, original_img = dataset[index]
    
    # 텐서를 numpy 배열로 변환하고 차원 순서 변경 (C,H,W) -> (H,W,C)
    blurred_img = blurred_img.permute(1, 2, 0).numpy()
    original_img = original_img.permute(1, 2, 0).numpy()
    
    # subplot 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 블러 이미지 표시
    ax1.imshow(blurred_img)
    ax1.set_title('Blurred Image')
    ax1.axis('off')
    
    # 원본 이미지 표시
    ax2.imshow(original_img)
    ax2.set_title('Original Image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# 데이터셋 생성
dataset = Cat_Dog_Dataset('dataset/training_set/training_set')

# 이미지 쌍 시각화 (인덱스 0의 이미지)
show_image_pair(dataset)

# 여러 이미지 쌍을 한번에 보고 싶다면:
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
for i in range(3):
    blurred_img, original_img = dataset[i]
    blurred_img = blurred_img.permute(1, 2, 0).numpy()
    original_img = original_img.permute(1, 2, 0).numpy()
    
    axes[i,0].imshow(blurred_img)
    axes[i,0].set_title(f'Blurred Image {i+1}')
    axes[i,0].axis('off')
    
    axes[i,1].imshow(original_img)
    axes[i,1].set_title(f'Original Image {i+1}')
    axes[i,1].axis('off')

    plt.imsave(f'blurred_image_{i+1}.png', blurred_img)
    plt.imsave(f'original_image_{i+1}.png', original_img)

plt.tight_layout()
plt.show()