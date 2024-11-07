import torch
from dataloader.Dataset import Cat_Dog_Dataset
import json
from model.Unet import UNet
from loss.ssim import SSIMLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config
with open('config/config.json') as f:
    config = json.load(f)

# dataset
train_dataset = Cat_Dog_Dataset(config['train']['data_dir'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)
validation_dataset = Cat_Dog_Dataset(config['validation']['data_dir'])
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['validation']['batch_size'], shuffle=False, num_workers=0)

# parameter
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss functions
l1_criterion = torch.nn.L1Loss()
ssim_criterion = SSIMLoss().to(device)

# training
best_val_loss = float('inf')
patience = 5  # early stopping patience
counter = 0

print("Training Start...")

for epoch in range(50):
    # Training phase
    model.train()
    train_loss = 0
    train_l1_loss = 0
    train_ssim_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate losses
        l1_loss = l1_criterion(outputs, labels)
        ssim_loss = ssim_criterion(outputs, labels)
        loss = l1_loss + 0.5 * ssim_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_l1_loss += l1_loss.item()
        train_ssim_loss += ssim_loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Step [{i+1}], "
                  f"Total Loss: {loss.item():.4f}, "
                  f"L1 Loss: {l1_loss.item():.4f}, "
                  f"SSIM Loss: {ssim_loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_l1_loss = train_l1_loss / len(train_loader)
    avg_train_ssim_loss = train_ssim_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_l1_loss = 0
    val_ssim_loss = 0
    
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            l1_loss = l1_criterion(outputs, labels)
            ssim_loss = ssim_criterion(outputs, labels)
            loss = l1_loss + 0.1 * ssim_loss
            
            val_loss += loss.item()
            val_l1_loss += l1_loss.item()
            val_ssim_loss += ssim_loss.item()
            
    avg_val_loss = val_loss / len(validation_loader)
    avg_val_l1_loss = val_l1_loss / len(validation_loader)
    avg_val_ssim_loss = val_ssim_loss / len(validation_loader)
    
    print(f"Epoch [{epoch+1}/50]")
    print(f"Train - Total Loss: {avg_train_loss:.4f}, L1 Loss: {avg_train_l1_loss:.4f}, SSIM Loss: {avg_train_ssim_loss:.4f}")
    print(f"Val - Total Loss: {avg_val_loss:.4f}, L1 Loss: {avg_val_l1_loss:.4f}, SSIM Loss: {avg_val_ssim_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, 'best_model.pth')
        print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
        counter = 0
    else:
        counter += 1
    
    # Early stopping
    if counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("Training finished!")

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.4f}")