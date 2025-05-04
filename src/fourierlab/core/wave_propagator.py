import torch
import numpy as np
from typing import Optional, Tuple, Dict
from .physics_solver import PhysicsSolver

class WavePropagator:
    """Advanced wave propagation with physics-based simulation."""
    
    def __init__(self,
                 grid_size: Tuple[int, int],
                 wavelength: float,
                 pixel_size: float,
                 device: str = 'cuda'):
        """
        Initialize the wave propagator.
        
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
        
        # Initialize physics solver
        self.physics_solver = PhysicsSolver(
            grid_size=grid_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=device
        )
        
        # Initialize coordinate grids
        self._init_grids()
    
    def _init_grids(self):
        """Initialize coordinate grids."""
        y, x = torch.meshgrid(
            torch.linspace(-self.grid_size[0]//2, self.grid_size[0]//2-1, self.grid_size[0]),
            torch.linspace(-self.grid_size[1]//2, self.grid_size[1]//2-1, self.grid_size[1]),
            indexing='ij'
        )
        self.x_grid = x * self.pixel_size
        self.y_grid = y * self.pixel_size
    
    def propagate(self,
                 field: torch.Tensor,
                 distance: float,
                 method: str = 'angular_spectrum',
                 medium: Optional[Dict] = None,
                 scatterers: Optional[list] = None) -> torch.Tensor:
        """
        Propagate the wave field with advanced physics simulation.
        
        Args:
            field: Input complex field
            distance: Propagation distance in meters
            method: Propagation method ('angular_spectrum' or 'rayleigh_sommerfeld')
            medium: Optional medium parameters (refractive_index, thickness)
            scatterers: Optional list of scatterer parameters
            
        Returns:
            Propagated complex field
        """
        # Add medium if specified
        if medium is not None:
            self.physics_solver.add_medium(
                refractive_index=medium['refractive_index'],
                thickness=medium['thickness']
            )
        
        # Add scatterers if specified
        if scatterers is not None:
            for scatterer in scatterers:
                self.physics_solver.add_scatterer(
                    position=scatterer['position'],
                    radius=scatterer['radius'],
                    refractive_index=scatterer['refractive_index']
                )
        
        # Propagate field
        return self.physics_solver.propagate(field, distance, method)
    
    def get_intensity(self, field: torch.Tensor) -> torch.Tensor:
        """Compute intensity from complex field."""
        return self.physics_solver.get_intensity(field)
    
    def add_medium(self, refractive_index: torch.Tensor, thickness: float):
        """Add a medium with varying refractive index."""
        self.physics_solver.add_medium(refractive_index, thickness)
    
    def add_scatterer(self, position: Tuple[float, float], radius: float, refractive_index: float):
        """Add a spherical scatterer to the simulation."""
        self.physics_solver.add_scatterer(position, radius, refractive_index) 