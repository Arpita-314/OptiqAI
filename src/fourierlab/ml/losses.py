import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FourierLoss(nn.Module):
    """Custom loss function combining MSE and Fourier domain metrics"""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2
    ):
        super().__init__()
        self.alpha = alpha  # Weight for spatial domain loss
        self.beta = beta   # Weight for magnitude spectrum loss
        self.gamma = gamma # Weight for phase spectrum loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Spatial domain loss (MSE)
        spatial_loss = F.mse_loss(pred, target)
        
        # Fourier domain losses
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Magnitude spectrum loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        magnitude_loss = F.mse_loss(pred_mag, target_mag)
        
        # Phase spectrum loss
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        # Combine losses
        total_loss = (
            self.alpha * spatial_loss +
            self.beta * magnitude_loss +
            self.gamma * phase_loss
        )
        
        return total_loss

class PhaseMaskLoss(nn.Module):
    """Loss function specifically for phase mask optimization"""
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        contrast_weight: float = 0.05
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.contrast_weight = contrast_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phase_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # Phase mask smoothness loss
        dx = phase_mask[:, :, 1:] - phase_mask[:, :, :-1]
        dy = phase_mask[:, 1:, :] - phase_mask[:, :-1, :]
        smoothness_loss = torch.mean(dx**2 + dy**2)
        
        # Contrast loss
        contrast_loss = -torch.mean(torch.abs(pred - target.mean()))
        
        # Total loss
        total_loss = (
            recon_loss +
            self.smoothness_weight * smoothness_loss +
            self.contrast_weight * contrast_loss
        )
        
        # Return loss and metrics
        metrics = {
            "reconstruction_loss": recon_loss.item(),
            "smoothness_loss": smoothness_loss.item(),
            "contrast_loss": contrast_loss.item()
        }
        
        return total_loss, metrics

class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) loss"""
    
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True
    ):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)
    
    def _gaussian(
        self,
        window_size: int,
        sigma: float
    ) -> torch.Tensor:
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2/float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss/gauss.sum()
    
    def _create_window(
        self,
        window_size: int
    ) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size).to(img1.device)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """Compute various metrics for evaluation"""
    # MSE
    mse = F.mse_loss(pred, target)
    
    # PSNR
    psnr = 10 * torch.log10(1.0 / mse)
    
    # SSIM
    ssim_loss = SSIMLoss()
    ssim = 1 - ssim_loss(pred, target)
    
    # Fourier domain metrics
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    
    # Magnitude spectrum similarity
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    mag_similarity = F.cosine_similarity(
        pred_mag.view(pred_mag.size(0), -1),
        target_mag.view(target_mag.size(0), -1)
    ).mean()
    
    # Phase spectrum similarity
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)
    phase_similarity = F.cosine_similarity(
        pred_phase.view(pred_phase.size(0), -1),
        target_phase.view(target_phase.size(0), -1)
    ).mean()
    
    return {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "magnitude_similarity": mag_similarity.item(),
        "phase_similarity": phase_similarity.item()
    } 