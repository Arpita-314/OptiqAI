import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ..physics.propagator import WavePropagator
import numpy as np

class PhaseCNN(nn.Module):
    """Lightweight CNN for phase retrieval with physical constraints"""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        n_channels: int = 1,
        n_filters: int = 32,
        n_layers: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize phase retrieval CNN
        
        Args:
            input_size: Input image size (height, width)
            n_channels: Number of input channels
            n_filters: Number of filters in first layer
            n_layers: Number of convolutional layers
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Calculate output size after convolutions
        self.conv_output_size = input_size[0] * input_size[1] * n_filters * (2**(n_layers-1))
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = n_channels
        out_channels = n_filters
        
        for i in range(n_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            ))
            in_channels = out_channels
            out_channels *= 2
        
        # Final convolution to get phase
        self.final_conv = nn.Conv2d(in_channels, 1, 1)
        
        # Initialize wave propagator for physical constraints
        self.propagator = WavePropagator()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)
            
        Returns:
            Phase tensor of shape (batch_size, 1, height, width)
        """
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Apply tanh to get phase in [-π, π]
        x = torch.tanh(x) * torch.pi
        
        return x
    
    def predict_phase(
        self,
        intensity: torch.Tensor,
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None
    ) -> torch.Tensor:
        """
        Predict phase from intensity with physical constraints
        
        Args:
            intensity: Intensity image tensor
            wavelength: Optional wavelength for physical constraints
            pixel_size: Optional pixel size for physical constraints
            
        Returns:
            Predicted phase
        """
        # Ensure input is normalized and finite
        intensity = torch.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0)
        intensity = intensity / (intensity.max() + 1e-8)
        
        # Add batch and channel dimensions if needed
        if intensity.dim() == 2:
            intensity = intensity.unsqueeze(0).unsqueeze(0)
        elif intensity.dim() == 3:
            intensity = intensity.unsqueeze(0)
        
        # Get phase prediction
        phase = self.forward(intensity)
        
        # Handle non-finite values
        phase = torch.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)
        
        # Apply physical constraints if parameters provided
        if wavelength is not None and pixel_size is not None:
            phase = self._apply_physical_constraints(phase, intensity, wavelength, pixel_size)
            # Handle any non-finite values after constraints
            phase = torch.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)
        
        return phase
    
    def _apply_physical_constraints(
        self,
        phase: torch.Tensor,
        intensity: torch.Tensor,
        wavelength: float,
        pixel_size: float
    ) -> torch.Tensor:
        """
        Apply physical constraints to phase prediction
        
        Args:
            phase: Predicted phase
            intensity: Input intensity
            wavelength: Wavelength of light
            pixel_size: Size of each pixel
            
        Returns:
            Phase with physical constraints applied
        """
        # Work with detached tensors
        phase = phase.detach().clone()
        
        # Calculate diffraction limit
        diffraction_limit = wavelength / (2 * pixel_size)
        
        # Apply low-pass filter to respect diffraction limit
        phase_fft = torch.fft.fft2(phase)
        freq_x = torch.fft.fftfreq(phase.shape[-2], pixel_size)
        freq_y = torch.fft.fftfreq(phase.shape[-1], pixel_size)
        FX, FY = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_mask = torch.sqrt(FX**2 + FY**2) <= diffraction_limit
        
        # Expand freq_mask to match phase_fft dimensions
        freq_mask = freq_mask.unsqueeze(0).unsqueeze(0)
        freq_mask = freq_mask.expand_as(phase_fft)
        
        phase_fft[~freq_mask] = 0
        phase = torch.abs(torch.fft.ifft2(phase_fft))
        
        # Ensure phase is in [-π, π]
        phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        
        # Calculate intensity from phase and compare with input
        field = torch.exp(1j * phase)
        predicted_intensity = self.propagator.calculate_intensity(field)
        
        # Normalize predicted intensity
        predicted_intensity = predicted_intensity / (predicted_intensity.max() + 1e-8)
        
        # Calculate intensity error
        intensity_error = torch.mean((predicted_intensity - intensity)**2)
        
        # If intensity error is too large, adjust phase using gradient descent
        if intensity_error > 0.1:
            lr = 0.01
            for _ in range(10):
                # Calculate gradient
                field = torch.exp(1j * phase)
                predicted_intensity = self.propagator.calculate_intensity(field)
                predicted_intensity = predicted_intensity / (predicted_intensity.max() + 1e-8)
                
                # Calculate error and gradient
                error = predicted_intensity - intensity
                gradient = 2 * error * predicted_intensity
                
                # Update phase
                phase = phase - lr * gradient
                
                # Project phase to [-π, π]
                phase = torch.atan2(torch.sin(phase), torch.cos(phase))
        
        return phase
    
    def save(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'n_channels': self.n_channels,
            'n_filters': self.n_filters,
            'n_layers': self.n_layers,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PhaseCNN':
        """Load model from file"""
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            n_channels=checkpoint['n_channels'],
            n_filters=checkpoint['n_filters'],
            n_layers=checkpoint['n_layers'],
            use_batch_norm=checkpoint['use_batch_norm'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 