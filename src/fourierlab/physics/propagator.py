import numpy as np
import torch
from typing import Union, Tuple, Optional

class WavePropagator:
    """Class for simulating optical wave propagation using angular spectrum method"""
    
    def __init__(self, wavelength: float = 632.8e-9, pixel_size: float = 5e-6):
        """
        Initialize wave propagator
        
        Args:
            wavelength: Wavelength of light in meters
            pixel_size: Size of each pixel in meters
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.k = 2 * np.pi / wavelength  # Wavenumber
    
    def angular_spectrum_propagate(
        self,
        field: Union[np.ndarray, torch.Tensor],
        distance: float,
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Propagate optical field using angular spectrum method
        
        Args:
            field: Complex field (amplitude * exp(i*phase))
            distance: Propagation distance in meters
            wavelength: Optional wavelength override
            pixel_size: Optional pixel size override
            
        Returns:
            Propagated complex field
        """
        # Update parameters if provided
        if wavelength is not None:
            self.wavelength = wavelength
            self.k = 2 * np.pi / wavelength
        if pixel_size is not None:
            self.pixel_size = pixel_size
        
        # Convert to numpy if tensor
        is_tensor = isinstance(field, torch.Tensor)
        if is_tensor:
            field = field.detach().cpu().numpy()
        
        # Get field dimensions
        nx, ny = field.shape
        
        # Calculate spatial frequencies
        dx = self.pixel_size
        fx = np.fft.fftfreq(nx, dx)
        fy = np.fft.fftfreq(ny, dx)
        FX, FY = np.meshgrid(fx, fy)
        
        # Calculate transfer function
        H = np.exp(1j * distance * np.sqrt(
            self.k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
        ))
        
        # Apply transfer function
        field_fft = np.fft.fft2(field)
        field_prop_fft = field_fft * H
        field_prop = np.fft.ifft2(field_prop_fft)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            field_prop = torch.from_numpy(field_prop)
        
        return field_prop
    
    def calculate_intensity(
        self,
        field: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Calculate intensity from complex field"""
        if isinstance(field, torch.Tensor):
            return torch.abs(field)**2
        return np.abs(field)**2
    
    def calculate_phase(
        self,
        field: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Calculate phase from complex field"""
        if isinstance(field, torch.Tensor):
            return torch.angle(field)
        return np.angle(field)
    
    def propagate_through_lens(
        self,
        field: Union[np.ndarray, torch.Tensor],
        focal_length: float,
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Simulate propagation through a thin lens
        
        Args:
            field: Input complex field
            focal_length: Focal length of lens in meters
            wavelength: Optional wavelength override
            pixel_size: Optional pixel size override
            
        Returns:
            Field after lens
        """
        # Update parameters if provided
        if wavelength is not None:
            self.wavelength = wavelength
            self.k = 2 * np.pi / wavelength
        if pixel_size is not None:
            self.pixel_size = pixel_size
        
        # Convert to numpy if tensor
        is_tensor = isinstance(field, torch.Tensor)
        if is_tensor:
            field = field.detach().cpu().numpy()
        
        # Get field dimensions
        nx, ny = field.shape
        
        # Calculate coordinates
        x = np.linspace(-nx//2, nx//2-1, nx) * self.pixel_size
        y = np.linspace(-ny//2, ny//2-1, ny) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        
        # Calculate lens phase
        lens_phase = np.exp(-1j * self.k * (X**2 + Y**2) / (2 * focal_length))
        
        # Apply lens phase
        field_after_lens = field * lens_phase
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            field_after_lens = torch.from_numpy(field_after_lens)
        
        return field_after_lens
    
    def calculate_psf(
        self,
        size: Tuple[int, int],
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculate point spread function
        
        Args:
            size: Size of PSF (nx, ny)
            wavelength: Optional wavelength override
            pixel_size: Optional pixel size override
            
        Returns:
            PSF intensity
        """
        # Create delta function
        field = np.zeros(size)
        field[size[0]//2, size[1]//2] = 1
        
        # Propagate
        field_prop = self.angular_spectrum_propagate(
            field,
            distance=1.0,  # Arbitrary distance
            wavelength=wavelength,
            pixel_size=pixel_size
        )
        
        # Calculate intensity
        return self.calculate_intensity(field_prop)
    
    def calculate_mtf(
        self,
        size: Tuple[int, int],
        wavelength: Optional[float] = None,
        pixel_size: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculate modulation transfer function
        
        Args:
            size: Size of MTF (nx, ny)
            wavelength: Optional wavelength override
            pixel_size: Optional pixel size override
            
        Returns:
            MTF
        """
        # Get PSF
        psf = self.calculate_psf(size, wavelength, pixel_size)
        
        # Calculate MTF
        if isinstance(psf, torch.Tensor):
            psf_fft = torch.fft.fft2(psf)
            mtf = torch.abs(psf_fft)
        else:
            psf_fft = np.fft.fft2(psf)
            mtf = np.abs(psf_fft)
        
        # Normalize
        mtf = mtf / mtf.max()
        
        return mtf 