import torch
import numpy as np
from typing import Optional, Tuple, Dict
import cupy as cp

class PhysicsSolver:
    """Advanced physics-based wave propagation solver using CUDA acceleration."""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 wavelength: float,
                 pixel_size: float,
                 device: str = 'cuda'):
        """
        Initialize the physics solver.
        
        Args:
            grid_size: Size of the simulation grid (height, width)
            wavelength: Wavelength of light in meters
            pixel_size: Size of each pixel in meters
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.grid_size = grid_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.device = device
        
        # Initialize CUDA grid
        self._init_cuda_grid()
        
        # Physical constants
        self.k = 2 * np.pi / wavelength  # Wave number
        self.dx = pixel_size
        self.dy = pixel_size
        
    def _init_cuda_grid(self):
        """Initialize CUDA grid for parallel computation."""
        if self.device == 'cuda':
            # Create coordinate grids
            y, x = cp.meshgrid(
                cp.linspace(-self.grid_size[0]//2, self.grid_size[0]//2-1, self.grid_size[0]),
                cp.linspace(-self.grid_size[1]//2, self.grid_size[1]//2-1, self.grid_size[1]),
                indexing='ij'
            )
            self.x_grid = x * self.dx
            self.y_grid = y * self.dy
            
            # Pre-compute transfer function
            self._compute_transfer_function()
    
    def _compute_transfer_function(self):
        """Compute the transfer function for wave propagation."""
        if self.device == 'cuda':
            # Spatial frequencies
            fx = cp.fft.fftfreq(self.grid_size[1], self.dx)
            fy = cp.fft.fftfreq(self.grid_size[0], self.dy)
            FX, FY = cp.meshgrid(fx, fy)
            
            # Transfer function
            k2 = (2 * cp.pi / self.wavelength) ** 2
            H = cp.exp(1j * cp.sqrt(k2 - (2 * cp.pi * FX) ** 2 - (2 * cp.pi * FY) ** 2))
            self.transfer_function = cp.fft.fftshift(H)
    
    def propagate(self, 
                 field: torch.Tensor,
                 distance: float,
                 method: str = 'angular_spectrum') -> torch.Tensor:
        """
        Propagate the wave field using physics-based methods.
        
        Args:
            field: Input complex field
            distance: Propagation distance in meters
            method: Propagation method ('angular_spectrum' or 'rayleigh_sommerfeld')
            
        Returns:
            Propagated complex field
        """
        if self.device == 'cuda':
            # Convert to CuPy array
            field_cp = cp.asarray(field.cpu().numpy())
            
            if method == 'angular_spectrum':
                # Angular spectrum method
                F = cp.fft.fft2(field_cp)
                F = cp.fft.fftshift(F)
                F *= self.transfer_function ** distance
                F = cp.fft.ifftshift(F)
                result = cp.fft.ifft2(F)
                
            elif method == 'rayleigh_sommerfeld':
                # Rayleigh-Sommerfeld diffraction
                r = cp.sqrt(self.x_grid**2 + self.y_grid**2 + distance**2)
                kernel = (1j * self.k - 1/r) * cp.exp(1j * self.k * r) / (2 * cp.pi * r**2)
                result = cp.fft.ifft2(cp.fft.fft2(field_cp) * cp.fft.fft2(kernel))
            
            # Convert back to PyTorch tensor
            return torch.from_numpy(cp.asnumpy(result)).to(self.device)
        else:
            raise NotImplementedError("CPU implementation not available")
    
    def add_medium(self, 
                  refractive_index: torch.Tensor,
                  thickness: float) -> None:
        """
        Add a medium with varying refractive index.
        
        Args:
            refractive_index: 2D tensor of refractive indices
            thickness: Thickness of the medium in meters
        """
        if self.device == 'cuda':
            self.refractive_index = cp.asarray(refractive_index.cpu().numpy())
            self.medium_thickness = thickness
            self._update_transfer_function()
    
    def _update_transfer_function(self):
        """Update transfer function to account for medium."""
        if hasattr(self, 'refractive_index'):
            # Update wave number based on local refractive index
            k_medium = self.k * self.refractive_index
            k2 = k_medium ** 2
            
            # Recompute transfer function
            fx = cp.fft.fftfreq(self.grid_size[1], self.dx)
            fy = cp.fft.fftfreq(self.grid_size[0], self.dy)
            FX, FY = cp.meshgrid(fx, fy)
            
            H = cp.exp(1j * cp.sqrt(k2 - (2 * cp.pi * FX) ** 2 - (2 * cp.pi * FY) ** 2))
            self.transfer_function = cp.fft.fftshift(H)
    
    def add_scatterer(self,
                     position: Tuple[float, float],
                     radius: float,
                     refractive_index: float) -> None:
        """
        Add a spherical scatterer to the simulation.
        
        Args:
            position: (x, y) position of scatterer in meters
            radius: Radius of scatterer in meters
            refractive_index: Refractive index of scatterer
        """
        if self.device == 'cuda':
            # Create scatterer mask
            x = self.x_grid - position[0]
            y = self.y_grid - position[1]
            r = cp.sqrt(x**2 + y**2)
            mask = r <= radius
            
            # Update refractive index
            if not hasattr(self, 'refractive_index'):
                self.refractive_index = cp.ones(self.grid_size)
            self.refractive_index[mask] = refractive_index
            
            self._update_transfer_function()
    
    def get_intensity(self, field: torch.Tensor) -> torch.Tensor:
        """Compute intensity from complex field."""
        if self.device == 'cuda':
            field_cp = cp.asarray(field.cpu().numpy())
            intensity = cp.abs(field_cp)**2
            return torch.from_numpy(cp.asnumpy(intensity)).to(self.device)
        else:
            return torch.abs(field)**2 