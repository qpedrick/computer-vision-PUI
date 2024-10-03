import ssl

import urllib

import torch
from torch import nn, optim
from torchvision import models
from dataset import RoadDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

ssl._create_default_https_context = ssl._create_unverified_context

# Check if GPU is available
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.segmentation.deeplabv3_resnet50(pretrained=True)

# Set model to train mode
model.train()

# Modify the classifier to match the number of classes (e.g., 1 class for road vs. non-road, or more)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=3, dilation=2, padding=2)

# Move the model to GPU if available
model = model.to(device)

# Loss function 
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


# Dataset and DataLoader
image_dir = "My_Kitti/train/images"
mask_dir = "My_Kitti/train/labels"
val_image_dir = "My_Kitti/validation/images"
val_mask_dir = "My_Kitti/validation/labels"
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

dataset = RoadDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=11, shuffle=True, drop_last=True)
valset = RoadDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=transform)
val_dataloader = DataLoader(valset, batch_size=5, shuffle=False, drop_last=True)

# Load the checkpoint
checkpoint = torch.load('deeplabv3_cycle_safety_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1  
min_val_loss = checkpoint['val_loss']

# Training loop
early_stop_counter = 0
num_epochs = 100  # Number of epochs
for epoch in range(start_epoch, num_epochs):
    print(epoch+1)
    batch_num = 0
    model.train()  
    running_loss = 0.0
    for images, masks in dataloader:
        batch_num += 1
        print(batch_num)
        # Move images and masks to GPU
        images = images.to(device)
        masks = masks.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)['out']

        # Compute the loss
        masks = masks.squeeze(1)  

        loss = criterion(outputs, masks.long())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_images, val_masks in val_dataloader:
            # Move validation data to GPU
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            # Forward pass
            val_outputs = model(val_images)['out']

            # Adjust mask dimensions
            val_masks = val_masks.squeeze(1)  # Remove channel dimension for masks

            # Compute validation loss
            val_loss += criterion(val_outputs, val_masks.long()).item()

    # Calculate average losses
    running_loss /= len(dataloader)
    val_loss /= len(val_dataloader)

    scheduler.step(val_loss)

    # Print losses for this epoch# Write dictionary to file
    with open("output.txt", "a") as file:
        file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

    if (val_loss > min_val_loss):
        early_stop_counter += 1

    if (early_stop_counter >= 3):
        print("Early Stop")
        break

    if (min_val_loss > val_loss):
        early_stop_counter = 0
        min_val_loss = val_loss
        # Save the checkpoint after every epoch
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }
        torch.save(checkpoint, f'deeplabv3_cycle_safety_model.pth')
        with open("output.txt", "a") as file:
            file.write(f"Model saved at epoch {epoch + 1}\n")
        print(f"Model saved at epoch {epoch + 1}")