import torch
from torchvision import models
from dataset import RoadDataset  # Custom dataset for testing
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the best model
model = models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=3, dilation=2, padding=2)
model.load_state_dict(torch.load('best_deeplabv3_model.pth'))  # Load the best model weights
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Testing Dataset and DataLoader
test_image_dir = "My_Kitti/images"
test_mask_dir = "My_Kitti/labels"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_dataset = RoadDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)

# Disable gradient calculations during testing
with torch.no_grad():
    running_iou = 0.0  # Track Intersection over Union (IoU) for segmentation
    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass: get model predictions
        outputs = model(images)['out']  
        predicted = torch.argmax(outputs, dim=1)  

        # Compute Intersection over Union (IoU) for evaluation
        intersection = (predicted & masks).float().sum((1, 2))  
        union = (predicted | masks).float().sum((1, 2))  
        iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
        running_iou += iou.mean().item()

    avg_iou = running_iou / len(test_loader)
    print(f'Average IoU: {avg_iou:.4f}')
