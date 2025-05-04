import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Union

class FourierCNN(nn.Module):
    """Enhanced CNN model for Fourier optics analysis with residual connections and attention"""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_blocks: int = 3,
        dropout_rate: float = 0.5,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction blocks
        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            self.blocks.append(
                FourierBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_attention=use_attention,
                    use_residual=use_residual
                )
            )
            in_ch = out_ch
        
        # Fourier layer
        self.fourier = FourierLayer()
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(in_ch)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.init_conv(x)
        
        # Feature extraction blocks
        for block in self.blocks:
            x = block(x)
        
        # Fourier transform
        x = self.fourier(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

class FourierBlock(nn.Module):
    """Block containing convolution, batch norm, and optional residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        
        if self.use_residual:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        if hasattr(self, 'attention'):
            out = self.attention(out)
        
        if self.use_residual:
            out += self.shortcut(identity)
        
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2)
        
        return out

class FourierLayer(nn.Module):
    """Enhanced Fourier transform layer with phase preservation"""
    
    def __init__(self, log_scale: bool = True):
        super().__init__()
        self.log_scale = log_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 2D FFT
        x_fft = torch.fft.fft2(x.float())
        
        # Get magnitude and phase
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Apply log scaling to magnitude if enabled
        if self.log_scale:
            magnitude = torch.log1p(magnitude)
        
        # Combine magnitude and phase
        x_complex = magnitude * torch.exp(1j * phase)
        
        return x_complex

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return x * torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class FourierUNet(nn.Module):
    """U-Net architecture with Fourier layers for optical field analysis"""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4
    ):
        super().__init__()
        
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.fourier_layers = nn.ModuleList()
        
        # Encoder
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            self.fourier_layers.append(FourierLayer())
            in_ch = out_ch
        
        # Decoder
        for i in range(depth-1, -1, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i-1)) if i > 0 else out_channels
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        features = []
        for i, (encoder, fourier) in enumerate(zip(self.encoder, self.fourier_layers)):
            x = encoder(x)
            x = fourier(x)
            features.append(x)
            if i < self.depth - 1:
                x = F.max_pool2d(x, 2)
        
        # Decoder path
        for i, decoder in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder(x)
        
        return x
