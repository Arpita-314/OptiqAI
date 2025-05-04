import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

class PhaseMaskGenerator:
    """Class for generating phase masks using optimization"""
    
    def __init__(self):
        self.wavelength = 632.8e-9  # meters
        self.pixel_size = 5e-6  # meters
        self.learning_rate = 0.01
        self.optimizer_type = 'adam'
        self.smoothness_weight = 0.1
        self.contrast_weight = 0.05
        
        self.phase_mask = None
        self.optimizer = None
        self.metrics = defaultdict(list)
    
    def set_parameters(self, wavelength=None, pixel_size=None, learning_rate=None,
                      optimizer_type=None, smoothness_weight=None, contrast_weight=None):
        """Set optimization parameters"""
        if wavelength is not None:
            self.wavelength = wavelength
        if pixel_size is not None:
            self.pixel_size = pixel_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if optimizer_type is not None:
            self.optimizer_type = optimizer_type
        if smoothness_weight is not None:
            self.smoothness_weight = smoothness_weight
        if contrast_weight is not None:
            self.contrast_weight = contrast_weight
    
    def _initialize_phase_mask(self, size):
        """Initialize random phase mask"""
        self.phase_mask = nn.Parameter(
            torch.rand(size, size, dtype=torch.float32) * 2 * np.pi - np.pi
        )
        
        # Initialize optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam([self.phase_mask], lr=self.learning_rate)
        else:  # sgd
            self.optimizer = optim.SGD([self.phase_mask], lr=self.learning_rate)
    
    def _propagate_field(self, field):
        """Propagate optical field through phase mask"""
        # Apply phase mask
        phase = torch.exp(1j * self.phase_mask)
        field = field * phase
        
        # Calculate propagation parameters
        dx = self.pixel_size
        k = 2 * np.pi / self.wavelength
        N = field.shape[0]
        
        # Create coordinate grids
        x = torch.linspace(-N//2, N//2-1, N) * dx
        y = torch.linspace(-N//2, N//2-1, N) * dx
        X, Y = torch.meshgrid(x, y)
        
        # Calculate transfer function
        H = torch.exp(1j * k * torch.sqrt(1 - (X**2 + Y**2) / (k**2 * dx**2)))
        H = torch.fft.fftshift(H)
        
        # Propagate field
        field_fft = torch.fft.fft2(field)
        field_prop = torch.fft.ifft2(field_fft * H)
        
        # Calculate intensity
        intensity = torch.abs(field_prop)**2
        return intensity / intensity.max()
    
    def _calculate_loss(self, target, output):
        """Calculate total loss"""
        # Mean squared error
        mse_loss = torch.mean((output - target)**2)
        
        # Smoothness loss
        smoothness_loss = torch.mean(
            (self.phase_mask[1:, :] - self.phase_mask[:-1, :])**2 +
            (self.phase_mask[:, 1:] - self.phase_mask[:, :-1])**2
        )
        
        # Contrast loss
        contrast_loss = -torch.mean(output * (1 - output))
        
        # Total loss
        total_loss = (mse_loss + 
                     self.smoothness_weight * smoothness_loss +
                     self.contrast_weight * contrast_loss)
        
        # Track metrics
        self.metrics['mse_loss'].append(mse_loss.item())
        self.metrics['smoothness_loss'].append(smoothness_loss.item())
        self.metrics['contrast_loss'].append(contrast_loss.item())
        self.metrics['total_loss'].append(total_loss.item())
        
        return total_loss
    
    def optimize(self, target, iterations=1000, callback=None):
        """Optimize phase mask"""
        # Convert target to tensor if needed
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        
        # Initialize phase mask
        size = target.shape[0]
        self._initialize_phase_mask(size)
        
        # Initial field
        field = torch.ones_like(target)
        
        # Optimization loop
        for i in range(iterations):
            self.optimizer.zero_grad()
            
            # Propagate field
            output = self._propagate_field(field)
            
            # Calculate loss
            loss = self._calculate_loss(target, output)
            
            # Backpropagate
            loss.backward()
            self.optimizer.step()
            
            # Callback
            if callback is not None:
                metrics = {
                    'mse_loss': self.metrics['mse_loss'][-1],
                    'smoothness_loss': self.metrics['smoothness_loss'][-1],
                    'contrast_loss': self.metrics['contrast_loss'][-1],
                    'total_loss': self.metrics['total_loss'][-1]
                }
                callback(i + 1, metrics)
        
        # Get final output
        final_output = self._propagate_field(field)
        
        return self.phase_mask.detach(), final_output.detach()
    
    def get_phase_mask(self):
        """Get current phase mask"""
        return self.phase_mask.detach() if self.phase_mask is not None else None
    
    def get_output(self):
        """Get current output"""
        if self.phase_mask is None:
            return None
        
        field = torch.ones_like(self.phase_mask)
        return self._propagate_field(field).detach()
    
    def get_metrics(self):
        """Get current metrics"""
        return dict(self.metrics) 