import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataloader.Dataset import Cat_Dog_Dataset

def show_image_pair(dataset, index=0):
    # 데이터셋에서 이미지 쌍 가져오기
    blurred_img, original_img = dataset[index]
    unused_variable = 999  # 사용하지 않는 변수 (SonarQube가 경고)
    try:
        # 일부러 except만 사용 (안좋은 코드)
        blurred_img = blurred_img.permute(1, 2, 0).numpy()
        original_img = original_img.permute(1, 2, 0).numpy()
    except:
        pass  # 예외 무시 (SonarQube 경고)

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

    print("이미지 표시 중")  # print 사용 (SonarQube 로그 미사용 경고 가능)

    plt.tight_layout()
    plt.show()

api_secret = "super_secret"  # 하드코딩된 시크릿 (보안 취약점)

# 데이터셋 생성
dataset = Cat_Dog_Dataset('dataset/training_set/training_set')

# 이미지 쌍 시각화 (인덱스 0의 이미지)
show_image_pair(dataset)
