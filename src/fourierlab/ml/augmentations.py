import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import random

class FourierAugmentation:
    """Data augmentation techniques specific to Fourier optics"""
    
    def __init__(
        self,
        phase_noise_std: float = 0.1,
        magnitude_noise_std: float = 0.05,
        rotation_range: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        flip_prob: float = 0.5
    ):
        self.phase_noise_std = phase_noise_std
        self.magnitude_noise_std = magnitude_noise_std
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
    
    def add_phase_noise(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Add random phase noise to the input"""
        # Convert to Fourier domain
        x_fft = torch.fft.fft2(x)
        
        # Add phase noise
        phase_noise = torch.randn_like(x) * self.phase_noise_std
        x_fft = x_fft * torch.exp(1j * phase_noise)
        
        # Convert back to spatial domain
        return torch.fft.ifft2(x_fft).real
    
    def add_magnitude_noise(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Add random magnitude noise to the input"""
        # Convert to Fourier domain
        x_fft = torch.fft.fft2(x)
        
        # Add magnitude noise
        magnitude_noise = torch.randn_like(x) * self.magnitude_noise_std
        x_fft = x_fft * (1 + magnitude_noise)
        
        # Convert back to spatial domain
        return torch.fft.ifft2(x_fft).real
    
    def random_rotation(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random rotation to the input"""
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        return F.interpolate(
            x,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=True
        )
    
    def random_scale(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random scaling to the input"""
        scale = random.uniform(*self.scale_range)
        return F.interpolate(
            x,
            scale_factor=scale,
            mode='bilinear',
            align_corners=True
        )
    
    def random_flip(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Apply random horizontal and vertical flips"""
        if random.random() < self.flip_prob:
            x = torch.flip(x, [-1])  # Horizontal flip
        if random.random() < self.flip_prob:
            x = torch.flip(x, [-2])  # Vertical flip
        return x
    
    def apply_augmentation(
        self,
        x: torch.Tensor,
        phase_noise: bool = True,
        magnitude_noise: bool = True,
        rotation: bool = True,
        scale: bool = True,
        flip: bool = True
    ) -> torch.Tensor:
        """Apply selected augmentations to the input"""
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
        return x

class FourierDataset(torch.utils.data.Dataset):
    """Dataset class with Fourier-specific augmentations"""
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        augmentation: Optional[FourierAugmentation] = None,
        transform: Optional[callable] = None
    ):
        self.data = data
        self.labels = labels
        self.augmentation = augmentation
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.augmentation is not None:
            x = self.augmentation.apply_augmentation(x)
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y

class FourierDataLoader:
    """Data loader with Fourier-specific augmentations"""
    
    def __init__(
        self,
        dataset: FourierDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self) -> int:
        return len(self.loader) 