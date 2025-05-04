import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
import random
import math
from .style_transfer import FourierStyleAugmentation

class AdvancedFourierAugmentation:
    """Advanced data augmentation techniques for Fourier optics"""
    
    def __init__(
        self,
        phase_noise_std: float = 0.1,
        magnitude_noise_std: float = 0.05,
        rotation_range: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        flip_prob: float = 0.5,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 0.1,
        cutout_prob: float = 0.5,
        cutout_size: Tuple[int, int] = (16, 16),
        mixup_alpha: float = 0.2,
        frequency_mask_prob: float = 0.3,
        frequency_mask_ratio: float = 0.1,
        style_prob: float = 0.3,
        content_weight: float = 1.0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6
    ):
        self.phase_noise_std = phase_noise_std
        self.magnitude_noise_std = magnitude_noise_std
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
        self.mixup_alpha = mixup_alpha
        self.frequency_mask_prob = frequency_mask_prob
        self.frequency_mask_ratio = frequency_mask_ratio
        
        # Initialize style transfer
        self.style_aug = FourierStyleAugmentation(
            style_prob=style_prob,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        )
    
    def elastic_deformation(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply elastic deformation to the input"""
        # Generate random displacement fields
        b, c, h, w = x.shape
        dx = torch.randn(b, 1, h, w) * self.elastic_alpha
        dy = torch.randn(b, 1, h, w) * self.elastic_alpha
        
        # Smooth displacement fields
        dx = F.avg_pool2d(dx, 3, stride=1, padding=1)
        dy = F.avg_pool2d(dy, 3, stride=1, padding=1)
        
        # Normalize coordinates
        y_coords = torch.arange(h, device=x.device).float()
        x_coords = torch.arange(w, device=x.device).float()
        y_coords = y_coords.view(1, 1, -1, 1).expand(b, 1, h, w)
        x_coords = x_coords.view(1, 1, 1, -1).expand(b, 1, h, w)
        
        # Apply displacement
        y_coords = y_coords + dy
        x_coords = x_coords + dx
        
        # Normalize to [-1, 1]
        y_coords = 2 * y_coords / (h - 1) - 1
        x_coords = 2 * x_coords / (w - 1) - 1
        
        # Stack coordinates
        coords = torch.stack([x_coords, y_coords], dim=-1)
        
        # Apply grid sampling
        return F.grid_sample(x, coords, mode='bilinear', align_corners=True)
    
    def cutout(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random cutout to the input"""
        if random.random() < self.cutout_prob:
            b, c, h, w = x.shape
            cutout_h, cutout_w = self.cutout_size
            
            # Random position
            top = random.randint(0, h - cutout_h)
            left = random.randint(0, w - cutout_w)
            
            # Apply cutout
            x[:, :, top:top + cutout_h, left:left + cutout_w] = 0
        
        return x
    
    def mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation"""
        # Generate mixing coefficient
        alpha = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Shuffle batch
        indices = torch.randperm(x.size(0))
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Mix samples
        x_mixed = alpha * x + (1 - alpha) * x_shuffled
        y_mixed = alpha * y + (1 - alpha) * y_shuffled
        
        return x_mixed, y_mixed
    
    def frequency_mask(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random frequency masking"""
        if random.random() < self.frequency_mask_prob:
            # Convert to Fourier domain
            x_fft = torch.fft.fft2(x)
            
            # Generate random mask
            b, c, h, w = x.shape
            mask = torch.ones_like(x_fft)
            mask_size = int(h * w * self.frequency_mask_ratio)
            
            # Random positions
            positions = torch.randperm(h * w)[:mask_size]
            mask.view(b, c, -1)[:, :, positions] = 0
            
            # Apply mask
            x_fft = x_fft * mask
            
            # Convert back to spatial domain
            x = torch.fft.ifft2(x_fft).real
        
        return x
    
    def random_erasing(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random erasing"""
        if random.random() < 0.5:
            b, c, h, w = x.shape
            
            # Random erasing area
            s = random.uniform(0.02, 0.4) * h * w
            r = random.uniform(0.3, 1/0.3)
            w_erase = int(math.sqrt(s * r))
            h_erase = int(math.sqrt(s / r))
            
            # Random position
            top = random.randint(0, h - h_erase)
            left = random.randint(0, w - w_erase)
            
            # Apply erasing
            x[:, :, top:top + h_erase, left:left + w_erase] = torch.randn_like(
                x[:, :, top:top + h_erase, left:left + w_erase]
            )
        
        return x
    
    def random_crop_and_resize(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Apply random crop and resize"""
        b, c, h, w = x.shape
        target_h, target_w = target_size
        
        # Random crop size
        crop_h = random.randint(target_h, h)
        crop_w = random.randint(target_w, w)
        
        # Random position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop and resize
        x = x[:, :, top:top + crop_h, left:left + crop_w]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        
        return x
    
    def apply_augmentation(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        style_image: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
        phase_noise: bool = True,
        magnitude_noise: bool = True,
        rotation: bool = True,
        scale: bool = True,
        flip: bool = True,
        elastic: bool = True,
        cutout: bool = True,
        mixup: bool = True,
        frequency_mask: bool = True,
        random_erasing: bool = True,
        crop_resize: bool = True,
        style_transfer: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply selected augmentations to the input"""
        # Basic augmentations
        if phase_noise:
            x = self.add_phase_noise(x)
        if magnitude_noise:
            x = self.add_magnitude_noise(x)
        if rotation:
            x = self.random_rotation(x)
        if scale:
            x = self.random_scale(x)
        if flip:
            x = self.random_flip(x)
        
        # Advanced augmentations
        if elastic:
            x = self.elastic_deformation(x)
        if cutout:
            x = self.cutout(x)
        if frequency_mask:
            x = self.frequency_mask(x)
        if random_erasing:
            x = self.random_erasing(x)
        if crop_resize and target_size is not None:
            x = self.random_crop_and_resize(x, target_size)
        
        # Style transfer
        if style_transfer and style_image is not None:
            x = self.style_aug.apply_style(x, style_image)
        
        # Mixup if labels are provided
        if mixup and y is not None:
            x, y = self.mixup(x, y)
            return x, y
        
        return x

class AdvancedFourierDataset(torch.utils.data.Dataset):
    """Dataset class with advanced Fourier-specific augmentations"""
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        style_images: Optional[torch.Tensor] = None,
        augmentation: Optional[AdvancedFourierAugmentation] = None,
        transform: Optional[callable] = None,
        target_size: Optional[Tuple[int, int]] = None
    ):
        self.data = data
        self.labels = labels
        self.style_images = style_images
        self.augmentation = augmentation
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        style = self.style_images[idx] if self.style_images is not None else None
        
        if self.augmentation is not None:
            result = self.augmentation.apply_augmentation(
                x.unsqueeze(0),
                y.unsqueeze(0) if y is not None else None,
                style.unsqueeze(0) if style is not None else None,
                self.target_size
            )
            if isinstance(result, tuple):
                x, y = result
                x = x.squeeze(0)
                y = y.squeeze(0)
            else:
                x = result.squeeze(0)
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y

class AdvancedFourierDataLoader:
    """Data loader with advanced Fourier-specific augmentations"""
    
    def __init__(
        self,
        dataset: AdvancedFourierDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self) -> int:
        return len(self.loader) 