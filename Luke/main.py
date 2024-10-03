import torch
import os
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=3, dilation=2, padding=2)
checkpoint = torch.load('deeplabv3_cycle_safety_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Transformation to apply to input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to match model input size
    transforms.ToTensor()           # Convert image to tensor
])

def overlay_mask_on_image(original_image, mask, alpha=0.5):
    """Overlay the mask on the original image with a given transparency (alpha)."""
    # Convert mask to binary mask (0 or 255 for visualization)
    mask = (mask * 255).astype(np.uint8)

    # Create a color version of the mask
    color_mask = np.zeros_like(original_image)
    color_mask[:, :, 1] = mask  # Set mask to green channel for better visualization

    # Overlay the mask on the original image with low opacity
    overlay = cv2.addWeighted(original_image, 1 - alpha, color_mask, alpha, 0)
    return overlay

def segment_image(image_path):
    # Load image using PIL
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out']  # Get model output
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Get the mask prediction

    # Convert the PIL image to a NumPy array for visualization
    original_image = np.array(image)

    # Overlay the mask on the original image
    overlaid_image = overlay_mask_on_image(original_image, mask, alpha=0.5)

    # Show the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image with Segmentation Mask")
    plt.imshow(overlaid_image)
    plt.axis('off')

    plt.show()

image_folder = 'My_Kitti/validation/images'

# Loop through all files in the directory
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    segment_image(image_path)
