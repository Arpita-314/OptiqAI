import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from ..physics.propagator import WavePropagator

class PatternGenerator:
    """Generator for various optical patterns"""
    
    def __init__(self):
        """Initialize pattern generator"""
        self.propagator = WavePropagator()
    
    def generate_pattern(
        self,
        pattern_type: str,
        size: int,
        width: int = 25,
        frequency: float = 10.0,
        wavelength: float = 632.8e-9,
        pixel_size: float = 5e-6,
        **kwargs
    ) -> np.ndarray:
        """
        Generate optical pattern
        
        Args:
            pattern_type: Type of pattern to generate
            size: Size of pattern (size x size)
            width: Width of pattern features
            frequency: Frequency of pattern (for periodic patterns)
            wavelength: Wavelength of light
            pixel_size: Size of each pixel
            **kwargs: Additional pattern-specific parameters
            
        Returns:
            Generated pattern
        """
        # Create coordinate grid
        x = np.linspace(-size//2, size//2-1, size)
        y = np.linspace(-size//2, size//2-1, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate pattern
        if pattern_type == "cross":
            pattern = self._generate_cross(X, Y, width)
        elif pattern_type == "circle":
            pattern = self._generate_circle(X, Y, width)
        elif pattern_type == "square":
            pattern = self._generate_square(X, Y, width)
        elif pattern_type == "grating":
            pattern = self._generate_grating(X, Y, frequency, width)
        elif pattern_type == "spiral":
            pattern = self._generate_spiral(X, Y, frequency, width)
        elif pattern_type == "zone_plate":
            pattern = self._generate_zone_plate(X, Y, frequency, width)
        elif pattern_type == "vortex":
            pattern = self._generate_vortex(X, Y, kwargs.get('order', 1))
        elif pattern_type == "bessel":
            pattern = self._generate_bessel(X, Y, frequency, width)
        elif pattern_type == "hermite":
            pattern = self._generate_hermite(X, Y, kwargs.get('n', 1), kwargs.get('m', 1))
        elif pattern_type == "laguerre":
            pattern = self._generate_laguerre(X, Y, kwargs.get('p', 0), kwargs.get('l', 1))
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Apply physical constraints
        pattern = self._apply_physical_constraints(pattern, wavelength, pixel_size)
        
        return pattern
    
    def _generate_cross(self, X: np.ndarray, Y: np.ndarray, width: int) -> np.ndarray:
        """Generate cross pattern"""
        pattern = np.zeros_like(X)
        pattern[np.abs(X) < width] = 1
        pattern[np.abs(Y) < width] = 1
        return pattern
    
    def _generate_circle(self, X: np.ndarray, Y: np.ndarray, width: int) -> np.ndarray:
        """Generate circle pattern"""
        R = np.sqrt(X**2 + Y**2)
        pattern = np.zeros_like(X)
        pattern[R < width] = 1
        return pattern
    
    def _generate_square(self, X: np.ndarray, Y: np.ndarray, width: int) -> np.ndarray:
        """Generate square pattern"""
        pattern = np.zeros_like(X)
        pattern[np.abs(X) < width] = 1
        pattern[np.abs(Y) < width] = 1
        pattern[np.abs(X) < width] &= pattern[np.abs(Y) < width]
        return pattern
    
    def _generate_grating(self, X: np.ndarray, Y: np.ndarray, frequency: float, width: int) -> np.ndarray:
        """Generate grating pattern"""
        pattern = np.zeros_like(X)
        grating = np.sin(2 * np.pi * frequency * X / X.max())
        pattern[np.abs(grating) < width/X.max()] = 1
        return pattern
    
    def _generate_spiral(self, X: np.ndarray, Y: np.ndarray, frequency: float, width: int) -> np.ndarray:
        """Generate spiral pattern"""
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        pattern = np.zeros_like(X)
        spiral = np.sin(2 * np.pi * frequency * R / R.max() + theta)
        pattern[np.abs(spiral) < width/R.max()] = 1
        return pattern
    
    def _generate_zone_plate(self, X: np.ndarray, Y: np.ndarray, frequency: float, width: int) -> np.ndarray:
        """Generate Fresnel zone plate pattern"""
        R = np.sqrt(X**2 + Y**2)
        pattern = np.zeros_like(X)
        zones = np.sin(np.pi * frequency * R**2 / R.max()**2)
        pattern[np.abs(zones) < width/R.max()] = 1
        return pattern
    
    def _generate_vortex(self, X: np.ndarray, Y: np.ndarray, order: int) -> np.ndarray:
        """Generate optical vortex pattern"""
        theta = np.arctan2(Y, X)
        pattern = np.exp(1j * order * theta)
        return np.abs(pattern)
    
    def _generate_bessel(self, X: np.ndarray, Y: np.ndarray, frequency: float, width: int) -> np.ndarray:
        """Generate Bessel beam pattern"""
        R = np.sqrt(X**2 + Y**2)
        pattern = np.zeros_like(X)
        bessel = np.abs(np.i0(2 * np.pi * frequency * R / R.max()))
        pattern[bessel > width/R.max()] = 1
        return pattern
    
    def _generate_hermite(self, X: np.ndarray, Y: np.ndarray, n: int, m: int) -> np.ndarray:
        """Generate Hermite-Gaussian beam pattern"""
        from scipy.special import hermite
        
        # Normalize coordinates
        x = X / X.max()
        y = Y / Y.max()
        
        # Generate Hermite polynomials
        Hn = hermite(n)(x)
        Hm = hermite(m)(y)
        
        # Generate Gaussian envelope
        gaussian = np.exp(-(x**2 + y**2))
        
        # Combine to form Hermite-Gaussian beam
        pattern = Hn * Hm * gaussian
        return np.abs(pattern)
    
    def _generate_laguerre(self, X: np.ndarray, Y: np.ndarray, p: int, l: int) -> np.ndarray:
        """Generate Laguerre-Gaussian beam pattern"""
        from scipy.special import genlaguerre
        
        # Convert to polar coordinates
        R = np.sqrt(X**2 + Y**2) / np.max(np.sqrt(X**2 + Y**2))
        theta = np.arctan2(Y, X)
        
        # Generate Laguerre polynomial
        L = genlaguerre(p, l)(2 * R**2)
        
        # Generate Gaussian envelope
        gaussian = np.exp(-R**2)
        
        # Combine to form Laguerre-Gaussian beam
        pattern = R**l * L * gaussian * np.exp(1j * l * theta)
        return np.abs(pattern)
    
    def _apply_physical_constraints(
        self,
        pattern: np.ndarray,
        wavelength: float,
        pixel_size: float
    ) -> np.ndarray:
        """
        Apply physical constraints to pattern
        
        Args:
            pattern: Input pattern
            wavelength: Wavelength of light
            pixel_size: Size of each pixel
            
        Returns:
            Pattern with physical constraints applied
        """
        # Calculate diffraction limit
        diffraction_limit = wavelength / (2 * pixel_size)
        
        # Apply low-pass filter to respect diffraction limit
        pattern_fft = np.fft.fft2(pattern)
        freq_x = np.fft.fftfreq(pattern.shape[0], pixel_size)
        freq_y = np.fft.fftfreq(pattern.shape[1], pixel_size)
        FX, FY = np.meshgrid(freq_x, freq_y)
        freq_mask = np.sqrt(FX**2 + FY**2) <= diffraction_limit
        pattern_fft[~freq_mask] = 0
        pattern = np.abs(np.fft.ifft2(pattern_fft))
        
        # Normalize safely
        pattern_min = pattern.min()
        pattern_max = pattern.max()
        if pattern_max > pattern_min:
            pattern = (pattern - pattern_min) / (pattern_max - pattern_min)
        else:
            pattern = np.zeros_like(pattern)
        
        return pattern 