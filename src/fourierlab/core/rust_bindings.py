import numpy as np
from typing import Optional, Tuple, Union, List
from .rust import WavePropagator as RustWavePropagator
from .rust import PatternGenerator as RustPatternGenerator
from .rust import PhaseMaskOptimizer as RustPhaseMaskOptimizer

class WavePropagator:
    """Wave propagation with SIMD optimization and batch processing."""
    
    def __init__(self, wavelength: float, pixel_size: float):
        """
        Initialize wave propagator.
        
        Args:
            wavelength: Wavelength in meters
            pixel_size: Pixel size in meters
        """
        self._propagator = RustWavePropagator(wavelength, pixel_size)
    
    def angular_spectrum(self, field: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate field using angular spectrum method.
        
        Args:
            field: Complex field to propagate
            distance: Propagation distance in meters
            
        Returns:
            Propagated complex field
        """
        return self._propagator.angular_spectrum(field, distance)
    
    def angular_spectrum_batch(self, fields: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate a batch of fields using angular spectrum method.
        
        Args:
            fields: Batch of complex fields to propagate (shape: [batch_size, height, width])
            distance: Propagation distance in meters
            
        Returns:
            Combined propagated complex field
        """
        return self._propagator.angular_spectrum_batch(fields, distance)
    
    def rayleigh_sommerfeld(self, field: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate field using Rayleigh-Sommerfeld method.
        
        Args:
            field: Complex field to propagate
            distance: Propagation distance in meters
            
        Returns:
            Propagated complex field
        """
        return self._propagator.rayleigh_sommerfeld(field, distance)
    
    def rayleigh_sommerfeld_batch(self, fields: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate a batch of fields using Rayleigh-Sommerfeld method.
        
        Args:
            fields: Batch of complex fields to propagate (shape: [batch_size, height, width])
            distance: Propagation distance in meters
            
        Returns:
            Combined propagated complex field
        """
        return self._propagator.rayleigh_sommerfeld_batch(fields, distance)
    
    def fresnel(self, field: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate field using Fresnel method.
        
        Args:
            field: Complex field to propagate
            distance: Propagation distance in meters
            
        Returns:
            Propagated complex field
        """
        return self._propagator.fresnel(field, distance)
    
    def fresnel_batch(self, fields: np.ndarray, distance: float) -> np.ndarray:
        """
        Propagate a batch of fields using Fresnel method.
        
        Args:
            fields: Batch of complex fields to propagate (shape: [batch_size, height, width])
            distance: Propagation distance in meters
            
        Returns:
            Combined propagated complex field
        """
        return self._propagator.fresnel_batch(fields, distance)

class PatternGenerator:
    """Pattern generation with SIMD optimization and batch processing."""
    
    def __init__(self, wavelength: float, pixel_size: float):
        """
        Initialize pattern generator.
        
        Args:
            wavelength: Wavelength in meters
            pixel_size: Pixel size in meters
        """
        self._generator = RustPatternGenerator(wavelength, pixel_size)
    
    def generate_pattern(
        self,
        pattern_type: str,
        size: Tuple[int, int],
        width: float,
        frequency: float,
        order: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a single pattern.
        
        Args:
            pattern_type: Type of pattern ("cross", "circle", "square", "grating", "spiral", "vortex", "bessel", "hermite", "laguerre")
            size: Pattern size (height, width)
            width: Pattern width parameter
            frequency: Pattern frequency parameter
            order: Order parameter for vortex, Hermite, and Laguerre patterns
            
        Returns:
            Generated complex pattern
        """
        return self._generator.generate_pattern(pattern_type, size, width, frequency, order)
    
    def generate_pattern_batch(
        self,
        pattern_type: str,
        size: Tuple[int, int],
        width: float,
        frequency: float,
        order: Optional[int] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Generate a batch of patterns.
        
        Args:
            pattern_type: Type of pattern ("cross", "circle", "square", "grating", "spiral", "vortex", "bessel", "hermite", "laguerre")
            size: Pattern size (height, width)
            width: Pattern width parameter
            frequency: Pattern frequency parameter
            order: Order parameter for vortex, Hermite, and Laguerre patterns
            batch_size: Number of patterns to generate
            
        Returns:
            Combined generated complex pattern
        """
        return self._generator.generate_pattern_batch(pattern_type, size, width, frequency, order, batch_size)

class PhaseMaskOptimizer:
    """Phase mask optimization with SIMD optimization and batch processing."""
    
    def __init__(
        self,
        wavelength: float,
        pixel_size: float,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
    ):
        """
        Initialize phase mask optimizer.
        
        Args:
            wavelength: Wavelength in meters
            pixel_size: Pixel size in meters
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of optimization iterations
        """
        self._optimizer = RustPhaseMaskOptimizer(wavelength, pixel_size, learning_rate, max_iterations)
    
    def optimize_phase_mask(
        self,
        target: np.ndarray,
        initial_phase: Optional[np.ndarray] = None,
        distance: float = 0.0,
    ) -> np.ndarray:
        """
        Optimize a single phase mask.
        
        Args:
            target: Target complex field
            initial_phase: Initial phase mask (optional)
            distance: Propagation distance in meters
            
        Returns:
            Optimized phase mask
        """
        return self._optimizer.optimize_phase_mask(target, initial_phase, distance)
    
    def optimize_phase_mask_batch(
        self,
        targets: np.ndarray,
        initial_phases: Optional[np.ndarray] = None,
        distance: float = 0.0,
    ) -> np.ndarray:
        """
        Optimize a batch of phase masks.
        
        Args:
            targets: Batch of target complex fields (shape: [batch_size, height, width])
            initial_phases: Batch of initial phase masks (shape: [batch_size, height, width], optional)
            distance: Propagation distance in meters
            
        Returns:
            Combined optimized phase mask
        """
        return self._optimizer.optimize_phase_mask_batch(targets, initial_phases, distance) 