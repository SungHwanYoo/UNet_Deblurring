import torch
import numpy as np
from dataloader.Dataset import Cat_Dog_Dataset
import json
from model.Unet import UNet
import math
from metric.psnr import PSNR
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration
with open('config/config.json') as f:
    config = json.load(f)

# Load test dataset
test_dataset = Cat_Dog_Dataset(config['test']['data_dir'])
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=config['test']['batch_size'], 
    shuffle=False, 
    num_workers=0
)

# Initialize model
model = UNet().to(device)

# Load best model
checkpoint = torch.load('weight/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test the model
total_psnr = 0
num_samples = 0

print("Starting evaluation...")

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate PSNR for each image in batch
        for j in range(outputs.size(0)):
            psnr = PSNR(outputs[j], labels[j])
            total_psnr += psnr.item()
            num_samples += 1
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1} batches")

# Calculate average PSNR
avg_psnr = total_psnr / num_samples

print(f"\nTest Results:")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Total number of test samples: {num_samples}")
print(f"Model used from epoch: {checkpoint['epoch']+1}")
print("Evaluation finished!")

if not os.path.exists('test_results'):
    os.makedirs('test_results')

num_samples_to_save = 1
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples_to_save:
            break
            
        images = images.to(device)
        outputs = model(images)
        
        # Save input, output, and target images
        for j in range(images.size(0)):
            # Convert tensors to numpy arrays and normalize
            input_image = images[j].cpu().permute(1, 2, 0).numpy()
            output_image = outputs[j].cpu().permute(1, 2, 0).numpy()
            target_image = labels[j].cpu().permute(1, 2, 0).numpy()
            
            # Normalize images to [0, 1] range
            input_image = np.clip(input_image, 0, 1)
            output_image = np.clip(output_image, 0, 1)
            target_image = np.clip(target_image, 0, 1)
            
            # Create figure with adjusted size and spacing
            plt.figure(figsize=(12, 3))  # 전체 figure 크기 조정
            
            # Plot blur image
            plt.subplot(131)  # 더 간단한 subplot 표기법 사용
            plt.imshow(input_image)
            plt.title('Blur Image', pad=10)  # title 여백 조정
            plt.axis('off')
            
            # Plot deblur result
            plt.subplot(132)
            plt.imshow(output_image)
            plt.title('Deblur Result', pad=10)
            plt.axis('off')
            
            # Plot original image
            plt.subplot(133)
            plt.imshow(target_image)
            plt.title('Original Image', pad=10)
            plt.axis('off')
            
            # Adjust layout with specific spacing
            plt.tight_layout(pad=1.5)  # 여백 조정
            
            # Save with higher dpi and reduced borders
            plt.savefig(f'test_results/sample_{i}_{j}.png', 
                       bbox_inches='tight',  # 불필요한 여백 제거
                       dpi=150)  # 해상도 조정
            plt.close()

print(f"Saved {num_samples_to_save} sample results in 'test_results' directory")