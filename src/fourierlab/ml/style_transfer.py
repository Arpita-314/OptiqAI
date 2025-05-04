import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import math

class FourierStyleTransfer:
    """Style transfer for Fourier optics using neural style transfer"""
    
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6,
        perceptual_weight: float = 1.0,
        num_steps: int = 300,
        style_layers: List[str] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        content_layers: List[str] = ['conv4_2']
    ):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.perceptual_weight = perceptual_weight
        self.num_steps = num_steps
        self.style_layers = style_layers
        self.content_layers = content_layers
        
        # Initialize VGG model for feature extraction
        self.vgg = self._build_vgg()
        self.vgg.eval()
        
        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss(
            layers=content_layers + style_layers
        )
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _build_vgg(self) -> nn.Module:
        """Build VGG model for feature extraction"""
        vgg = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        return vgg
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def _content_loss(
        self,
        content_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute content loss"""
        loss = 0
        for layer in self.content_layers:
            loss += F.mse_loss(
                content_features[layer],
                target_features[layer]
            )
        return loss
    
    def _style_loss(
        self,
        style_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute style loss"""
        loss = 0
        for layer in self.style_layers:
            style_gram = self._gram_matrix(style_features[layer])
            target_gram = self._gram_matrix(target_features[layer])
            loss += F.mse_loss(style_gram, target_gram)
        return loss
    
    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss"""
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)
    
    def _extract_features(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract features from VGG model"""
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if f'conv{i+1}' in self.style_layers or f'conv{i+1}' in self.content_layers:
                features[f'conv{i+1}'] = x
        return features
    
    def transfer_style(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Transfer style from style image to content image"""
        # Move models and inputs to device
        self.vgg = self.vgg.to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        content = content.to(device)
        style = style.to(device)
        
        # Initialize output image
        output = content.clone().requires_grad_(True)
        
        # Extract features
        content_features = self._extract_features(content)
        style_features = self._extract_features(style)
        
        # Optimize output image
        optimizer = torch.optim.LBFGS([output])
        
        metrics = {
            'content_loss': [],
            'style_loss': [],
            'tv_loss': [],
            'perceptual_loss': [],
            'total_loss': []
        }
        
        def closure():
            optimizer.zero_grad()
            
            # Extract features
            output_features = self._extract_features(output)
            
            # Compute losses
            content_loss = self._content_loss(output_features, content_features)
            style_loss = self._style_loss(output_features, style_features)
            tv_loss = self._total_variation_loss(output)
            perceptual_loss = self.perceptual_loss(output, content)
            
            # Total loss
            total_loss = (
                self.content_weight * content_loss +
                self.style_weight * style_loss +
                self.tv_weight * tv_loss +
                self.perceptual_weight * perceptual_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update metrics
            metrics['content_loss'].append(content_loss.item())
            metrics['style_loss'].append(style_loss.item())
            metrics['tv_loss'].append(tv_loss.item())
            metrics['perceptual_loss'].append(perceptual_loss.item())
            metrics['total_loss'].append(total_loss.item())
            
            return total_loss
        
        # Run optimization
        for step in range(self.num_steps):
            optimizer.step(closure)
        
        # Average metrics
        for k in metrics:
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        
        return output.detach(), metrics
    
    def transfer_style_batch(
        self,
        content_batch: torch.Tensor,
        style_batch: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Transfer style for a batch of images"""
        outputs = []
        metrics = {
            'content_loss': 0.0,
            'style_loss': 0.0,
            'tv_loss': 0.0,
            'perceptual_loss': 0.0,
            'total_loss': 0.0
        }
        
        for content, style in zip(content_batch, style_batch):
            output, batch_metrics = self.transfer_style(
                content.unsqueeze(0),
                style.unsqueeze(0),
                device
            )
            outputs.append(output)
            
            # Update metrics
            for k in metrics:
                metrics[k] += batch_metrics[k]
        
        # Average metrics
        for k in metrics:
            metrics[k] /= len(content_batch)
        
        return torch.cat(outputs, dim=0), metrics

class FourierStyleAugmentation:
    """Style augmentation for Fourier optics"""
    
    def __init__(
        self,
        style_transfer: Optional[FourierStyleTransfer] = None,
        style_prob: float = 0.3,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6
    ):
        self.style_transfer = style_transfer or FourierStyleTransfer(
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        )
        self.style_prob = style_prob
    
    def apply_style(
        self,
        x: torch.Tensor,
        style_image: torch.Tensor
    ) -> torch.Tensor:
        """Apply style transfer to input"""
        if torch.rand(1) < self.style_prob:
            output, _ = self.style_transfer.transfer_style(x, style_image)
            return output
        return x
    
    def apply_style_batch(
        self,
        x: torch.Tensor,
        style_images: torch.Tensor
    ) -> torch.Tensor:
        """Apply style transfer to batch"""
        if torch.rand(1) < self.style_prob:
            output, _ = self.style_transfer.transfer_style_batch(x, style_images)
            return output
        return x

class FastNeuralStyle(nn.Module):
    """Fast Neural Style Transfer network"""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 64,
        num_blocks: int = 5,
        use_instance_norm: bool = True
    ):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, 9, 1, 4)
        self.in1 = nn.InstanceNorm2d(num_filters) if use_instance_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        
        # Downsampling
        self.down1 = nn.Conv2d(num_filters, num_filters*2, 3, 2, 1)
        self.in2 = nn.InstanceNorm2d(num_filters*2) if use_instance_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
        self.down2 = nn.Conv2d(num_filters*2, num_filters*4, 3, 2, 1)
        self.in3 = nn.InstanceNorm2d(num_filters*4) if use_instance_norm else nn.Identity()
        self.relu3 = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters*4, use_instance_norm)
            for _ in range(num_blocks)
        ])
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(num_filters*4, num_filters*2, 3, 2, 1, 1)
        self.in4 = nn.InstanceNorm2d(num_filters*2) if use_instance_norm else nn.Identity()
        self.relu4 = nn.ReLU(inplace=True)
        
        self.up2 = nn.ConvTranspose2d(num_filters*2, num_filters, 3, 2, 1, 1)
        self.in5 = nn.InstanceNorm2d(num_filters) if use_instance_norm else nn.Identity()
        self.relu5 = nn.ReLU(inplace=True)
        
        # Output convolution
        self.conv2 = nn.Conv2d(num_filters, in_channels, 9, 1, 4)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.relu1(self.in1(self.conv1(x)))
        
        # Downsampling
        x = self.relu2(self.in2(self.down1(x)))
        x = self.relu3(self.in3(self.down2(x)))
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Upsampling
        x = self.relu4(self.in4(self.up1(x)))
        x = self.relu5(self.in5(self.up2(x)))
        
        # Output
        x = self.tanh(self.conv2(x))
        return x

class ResidualBlock(nn.Module):
    """Residual block for Fast Neural Style"""
    
    def __init__(self, channels: int, use_instance_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(channels) if use_instance_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2 = nn.InstanceNorm2d(channels) if use_instance_norm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu1(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        return x + residual

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor
    ) -> torch.Tensor:
        # Calculate mean and variance
        content_mean = torch.mean(content, dim=[2, 3], keepdim=True)
        content_var = torch.var(content, dim=[2, 3], keepdim=True) + self.epsilon
        style_mean = torch.mean(style, dim=[2, 3], keepdim=True)
        style_var = torch.var(style, dim=[2, 3], keepdim=True) + self.epsilon
        
        # Normalize content
        content_norm = (content - content_mean) / torch.sqrt(content_var)
        
        # Apply style
        return style_mean + torch.sqrt(style_var) * content_norm

class AdaINStyleTransfer:
    """Style transfer using Adaptive Instance Normalization"""
    
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        num_steps: int = 300,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6,
        perceptual_weight: float = 1.0
    ):
        self.encoder = encoder or self._build_encoder()
        self.decoder = decoder or self._build_decoder()
        self.num_steps = num_steps
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss()
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder network"""
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network"""
        return nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
    
    def _content_loss(
        self,
        content_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute content loss"""
        return F.mse_loss(content_features, target_features)
    
    def _style_loss(
        self,
        style_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute style loss using AdaIN"""
        adain = AdaIN()
        return F.mse_loss(
            adain(target_features, style_features),
            target_features
        )
    
    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss"""
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (tv_h + tv_w) / (b * c * h * w)
    
    def transfer_style(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Transfer style using AdaIN"""
        # Move models and inputs to device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        content = content.to(device)
        style = style.to(device)
        
        # Extract features
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        
        # Initialize output
        output = content.clone().requires_grad_(True)
        
        # Optimize output
        optimizer = torch.optim.LBFGS([output])
        
        metrics = {
            'content_loss': [],
            'style_loss': [],
            'tv_loss': [],
            'perceptual_loss': [],
            'total_loss': []
        }
        
        def closure():
            optimizer.zero_grad()
            
            # Extract features
            output_features = self.encoder(output)
            
            # Compute losses
            content_loss = self._content_loss(output_features, content_features)
            style_loss = self._style_loss(output_features, style_features)
            tv_loss = self._total_variation_loss(output)
            perceptual_loss = self.perceptual_loss(output, content)
            
            # Total loss
            total_loss = (
                self.content_weight * content_loss +
                self.style_weight * style_loss +
                self.tv_weight * tv_loss +
                self.perceptual_weight * perceptual_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Update metrics
            metrics['content_loss'].append(content_loss.item())
            metrics['style_loss'].append(style_loss.item())
            metrics['tv_loss'].append(tv_loss.item())
            metrics['perceptual_loss'].append(perceptual_loss.item())
            metrics['total_loss'].append(total_loss.item())
            
            return total_loss
        
        # Run optimization
        for step in range(self.num_steps):
            optimizer.step(closure)
        
        # Average metrics
        for k in metrics:
            metrics[k] = sum(metrics[k]) / len(metrics[k])
        
        return output.detach(), metrics

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(
        self,
        layers: List[str] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        self.vgg = self._build_vgg()
        self.vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _build_vgg(self) -> nn.Module:
        """Build VGG model for feature extraction"""
        vgg = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        return vgg
    
    def _extract_features(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract features from VGG model"""
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if f'conv{i+1}' in self.layers:
                features[f'conv{i+1}'] = x
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss"""
        x_features = self._extract_features(x)
        y_features = self._extract_features(y)
        
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            loss += weight * F.mse_loss(x_features[layer], y_features[layer])
        
        return loss 