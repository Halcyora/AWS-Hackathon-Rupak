import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import os
from physics_engine import calculate_fractal_dimension
from model import FractalHybridCNN

import random
from torch.utils.data import Subset

# Custom Dataset to include Fractal Dimension calculation on the fly
class FractalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_pil, label = self.data[idx]
        
        # Convert PIL to CV2 grayscale array for Physics Engine
        img_cv2 = np.array(img_pil.convert('L'))
        fd_value = calculate_fractal_dimension(img_cv2)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
            
        return img_tensor, torch.tensor(fd_value, dtype=torch.float32), label

def train_model():
    print("Initializing proper ML pipeline...")
    
    # Preprocessing for MobileNetV3
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # MobileNet expects 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data (Point this to your extracted Kaggle folder)
    # FOR TIME: Only use a subset if CPU training is too slow
    train_dataset = FractalDataset(root_dir='chest_xray/train', transform=transform)

    # Shuffle and grab exactly 500 images so it trains in minutes, not hours
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:500]
    
    train_subset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = FractalHybridCNN(num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training on {device}... This will take time.")
    epochs = 3 # Keep it low for the hackathon sprint!
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, fds, labels) in enumerate(train_loader):
            images, fds, labels = images.to(device), fds.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, fds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}")
                running_loss = 0.0

    print("Training complete. Saving weights...")
    torch.save(model.state_dict(), 'fractallens_weights.pth')
    print("Model saved as fractallens_weights.pth")

if __name__ == "__main__":
    train_model()