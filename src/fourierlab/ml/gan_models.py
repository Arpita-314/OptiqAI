import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class FourierGenerator(nn.Module):
    """Generator network for phase retrieval GAN"""
    
    def __init__(
        self,
        latent_dim: int = 100,
        out_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Initial dense layer
        self.dense = nn.Linear(latent_dim, 4 * 4 * base_channels * 8)
        
        # Convolutional blocks
        self.blocks = nn.ModuleList()
        in_ch = base_channels * 8
        for i in range(num_blocks):
            out_ch = in_ch // 2
            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = out_ch
        
        # Final convolution
        self.final = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        
        # Fourier layer
        self.fourier = FourierLayer()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Initial dense layer
        x = self.dense(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        
        # Convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = self.final(x)
        
        # Apply Fourier transform
        x = self.fourier(x)
        
        return torch.tanh(x)

class FourierDiscriminator(nn.Module):
    """Discriminator network for phase retrieval GAN"""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Convolutional blocks
        self.blocks = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_blocks - 1):
            out_ch = in_ch * 2
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch
        
        # Final convolution
        self.final = nn.Conv2d(in_ch, 1, 4, 1, 0)
        
        # Fourier layer
        self.fourier = FourierLayer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Fourier transform
        x = self.fourier(x)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Convolutional blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        x = self.final(x)
        
        return x.view(-1, 1).squeeze(1)

class FourierGAN(nn.Module):
    """GAN model for phase retrieval"""
    
    def __init__(
        self,
        latent_dim: int = 100,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.generator = FourierGenerator(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
        
        self.discriminator = FourierDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

class FourierGANLoss:
    """Loss functions for Fourier GAN"""
    
    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_phase: float = 10.0,
        lambda_magnitude: float = 5.0
    ):
        self.lambda_l1 = lambda_l1
        self.lambda_phase = lambda_phase
        self.lambda_magnitude = lambda_magnitude
        
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
    
    def generator_loss(
        self,
        fake_outputs: torch.Tensor,
        generated_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Adversarial loss
        g_loss = self.bce(fake_outputs, torch.ones_like(fake_outputs))
        
        # L1 loss
        l1_loss = self.l1(generated_images, target_images)
        
        # Phase loss
        phase_loss = self._phase_loss(generated_images, target_images)
        
        # Magnitude loss
        magnitude_loss = self._magnitude_loss(generated_images, target_images)
        
        # Total loss
        total_loss = (
            g_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_phase * phase_loss +
            self.lambda_magnitude * magnitude_loss
        )
        
        metrics = {
            'g_loss': g_loss.item(),
            'l1_loss': l1_loss.item(),
            'phase_loss': phase_loss.item(),
            'magnitude_loss': magnitude_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def discriminator_loss(
        self,
        real_outputs: torch.Tensor,
        fake_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Real loss
        real_loss = self.bce(real_outputs, torch.ones_like(real_outputs))
        
        # Fake loss
        fake_loss = self.bce(fake_outputs, torch.zeros_like(fake_outputs))
        
        # Total loss
        total_loss = real_loss + fake_loss
        
        metrics = {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def _phase_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Convert to Fourier domain
        gen_fft = torch.fft.fft2(generated)
        target_fft = torch.fft.fft2(target)
        
        # Get phases
        gen_phase = torch.angle(gen_fft)
        target_phase = torch.angle(target_fft)
        
        # Compute phase difference
        phase_diff = torch.abs(gen_phase - target_phase)
        phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
        
        return torch.mean(phase_diff)
    
    def _magnitude_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Convert to Fourier domain
        gen_fft = torch.fft.fft2(generated)
        target_fft = torch.fft.fft2(target)
        
        # Get magnitudes
        gen_mag = torch.abs(gen_fft)
        target_mag = torch.abs(target_fft)
        
        # Compute magnitude difference
        return F.mse_loss(gen_mag, target_mag)

class FourierGANOptimizer:
    """Optimizer for Fourier GAN"""
    
    def __init__(
        self,
        gan: FourierGAN,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999
    ):
        self.gan = gan
        
        # Generator optimizer
        self.optimizer_g = torch.optim.Adam(
            gan.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        
        # Discriminator optimizer
        self.optimizer_d = torch.optim.Adam(
            gan.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
    
    def step(
        self,
        g_loss: torch.Tensor,
        d_loss: torch.Tensor
    ) -> None:
        # Generator step
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()
        
        # Discriminator step
        self.optimizer_d.zero_grad()
        d_loss.backward()
        self.optimizer_d.step()
    
    def get_lr(self) -> Tuple[float, float]:
        """Get current learning rates"""
        return (
            self.optimizer_g.param_groups[0]['lr'],
            self.optimizer_d.param_groups[0]['lr']
        ) 