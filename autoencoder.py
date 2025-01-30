import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the custom dataset
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        if self.transform:
            image = self.transform(image)
        return image

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (240, 320, 3) -> (120, 160, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (120, 160, 64) -> (60, 80, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (60, 80, 128) -> (30, 40, 256)
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # (30, 40, 256) -> (60, 80, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),  # (60, 80, 128) -> (120, 160, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),  # (120, 160, 64) -> (240, 320, 3)
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training script
def train_autoencoder(folder_path, batch_size=32, epochs=20, learning_rate=1e-3):
    # Set up transformations
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = ImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model, Loss, Optimizer
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
    
    # Save the model
    torch.save(model.state_dict(), "autoencoder.pth")
    print("Model saved as autoencoder.pth")

# Train the autoencoder
if __name__ == "__main__":
    folder_path = "path/to/your/images"  # Replace with the actual path
    train_autoencoder(folder_path)
