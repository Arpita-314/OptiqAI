import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class FourierTransformer(nn.Module):
    """Transformer model for Fourier optics analysis"""
    
    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_fourier: bool = True
    ):
        super().__init__()
        
        self.use_fourier = use_fourier
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fourier layer
        if use_fourier:
            self.fourier = FourierLayer()
        
        # Output projection
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Reshape back
        x = x.permute(0, 2, 1).view(b, c, h, w)
        
        # Apply Fourier transform if enabled
        if self.use_fourier:
            x = self.fourier(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class FourierTransformerUNet(nn.Module):
    """U-Net architecture with Transformer blocks for optical field analysis"""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        self.fourier_layers = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(4):  # 4 levels
            out_ch = d_model // (2 ** (3-i))
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            
            # Transformer block
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=out_ch,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_blocks.append(
                nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
            )
            
            # Fourier layer
            self.fourier_layers.append(FourierLayer())
            
            in_ch = out_ch
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(3):  # 3 levels
            in_ch = d_model // (2 ** (2-i))
            out_ch = d_model // (2 ** (3-i))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch * 2, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(d_model // 8, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        features = []
        for encoder, transformer, fourier in zip(
            self.encoder, self.transformer_blocks, self.fourier_layers
        ):
            x = encoder(x)
            
            # Apply transformer
            b, c, h, w = x.shape
            x_t = x.flatten(2).permute(0, 2, 1)
            x_t = transformer(x_t)
            x = x_t.permute(0, 2, 1).view(b, c, h, w)
            
            # Apply Fourier transform
            x = fourier(x)
            
            features.append(x)
            if len(features) < 4:  # Don't downsample after last encoder block
                x = F.max_pool2d(x, 2)
        
        # Decoder path
        for i, decoder in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, features[-(i+2)]], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class FourierLayer(nn.Module):
    """Fourier transform layer with phase preservation"""
    
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