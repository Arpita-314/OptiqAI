import cupy as cp
import numpy as np
from numba import cuda
import torch

@cuda.jit(device=True)
def cuda_transfer_function(kx, ky, wavelength, distance):
    """Compute the transfer function for angular spectrum propagation on GPU."""
    k = 2 * cp.pi / wavelength
    k_squared = k**2 - (2*cp.pi*kx)**2 - (2*cp.pi*ky)**2
    # Handle evanescent waves
    if k_squared < 0:
        return 0.0
    return cp.exp(1j * distance * cp.sqrt(k_squared))

class GPUWavePropagator:
    def __init__(self, device=None):
        """Initialize GPU propagator.
        
        Args:
            device: CUDA device to use (None for auto-select)
        """
        if device is not None:
            self.device = cp.cuda.Device(device)
        else:
            self.device = cp.cuda.Device(0)
        
        # Initialize memory pool
        self.memory_pool = cp.get_default_memory_pool()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - free GPU memory."""
        self.memory_pool.free_all_blocks()
        
    def angular_spectrum(self, field, wavelength, distance):
        """Propagate field using angular spectrum method on GPU.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            
        Returns:
            Propagated field (numpy array)
        """
        with self.device:
            # Move data to GPU
            field_gpu = cp.asarray(field)
            
            # Compute FFT
            fft_gpu = cp.fft.fft2(field_gpu)
            
            # Generate frequency grid
            kx = cp.fft.fftfreq(field.shape[0])
            ky = cp.fft.fftfreq(field.shape[1])
            kx_grid, ky_grid = cp.meshgrid(kx, ky)
            
            # Compute transfer function
            k = 2 * cp.pi / wavelength
            k_squared = k**2 - (2*cp.pi*kx_grid)**2 - (2*cp.pi*ky_grid)**2
            mask = k_squared > 0
            transfer = cp.zeros_like(k_squared, dtype=cp.complex128)
            transfer[mask] = cp.exp(1j * distance * cp.sqrt(k_squared[mask]))
            
            # Apply and inverse FFT
            result_gpu = cp.fft.ifft2(fft_gpu * transfer)
            
            # Return to CPU
            return cp.asnumpy(result_gpu)
            
    def rayleigh_sommerfeld(self, field, wavelength, distance):
        """Propagate field using Rayleigh-Sommerfeld diffraction on GPU.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            
        Returns:
            Propagated field (numpy array)
        """
        with self.device:
            # Move data to GPU
            field_gpu = cp.asarray(field)
            
            # Compute FFT
            fft_gpu = cp.fft.fft2(field_gpu)
            
            # Generate frequency grid
            kx = cp.fft.fftfreq(field.shape[0])
            ky = cp.fft.fftfreq(field.shape[1])
            kx_grid, ky_grid = cp.meshgrid(kx, ky)
            
            # Compute transfer function for Rayleigh-Sommerfeld
            k = 2 * cp.pi / wavelength
            k_squared = k**2 - (2*cp.pi*kx_grid)**2 - (2*cp.pi*ky_grid)**2
            mask = k_squared > 0
            transfer = cp.zeros_like(k_squared, dtype=cp.complex128)
            transfer[mask] = cp.exp(1j * distance * cp.sqrt(k_squared[mask]))
            
            # Apply and inverse FFT
            result_gpu = cp.fft.ifft2(fft_gpu * transfer)
            
            # Return to CPU
            return cp.asnumpy(result_gpu)
            
    def fresnel(self, field, wavelength, distance, dx, dy):
        """Propagate field using Fresnel diffraction on GPU.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            dx: Pixel size in x direction (meters)
            dy: Pixel size in y direction (meters)
            
        Returns:
            Propagated field (numpy array)
        """
        with self.device:
            # Move data to GPU
            field_gpu = cp.asarray(field)
            
            # Generate spatial frequency grid
            nx, ny = field.shape
            kx = cp.fft.fftfreq(nx) / dx
            ky = cp.fft.fftfreq(ny) / dy
            kx_grid, ky_grid = cp.meshgrid(kx, ky)
            
            # Compute Fresnel transfer function
            k = 2 * cp.pi / wavelength
            transfer = cp.exp(1j * distance * k * (1 - (wavelength * kx_grid)**2 - (wavelength * ky_grid)**2))
            
            # Apply FFT, transfer function, and inverse FFT
            fft_gpu = cp.fft.fft2(field_gpu)
            result_gpu = cp.fft.ifft2(fft_gpu * transfer)
            
            # Return to CPU
            return cp.asnumpy(result_gpu) 