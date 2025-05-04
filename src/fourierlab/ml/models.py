import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierCNN(nn.Module):
    """CNN model for Fourier optics analysis"""
    
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourier layer
        self.fourier = FourierLayer()
        
        # Classification layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted for 32x32 input after pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 128x128
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 64x64
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 32x32
        
        # Fourier transform
        x = self.fourier(x)
        
        # Adaptive pooling to ensure consistent size
        x = F.adaptive_avg_pool2d(x, (4, 4))
        
        # Classification
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FourierLayer(nn.Module):
    """Custom layer for Fourier transform operations"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Apply 2D FFT
        x_fft = torch.fft.fft2(x.float())
        # Get magnitude spectrum
        x_magnitude = torch.abs(x_fft)
        # Log scale for better feature representation
        x_log = torch.log1p(x_magnitude)
        return x_log
