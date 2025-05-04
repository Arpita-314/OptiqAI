import numpy as np
import pytest
from ..physics.propagator import WavePropagator

def test_propagator_conserves_energy():
    """Test that wave propagation conserves energy"""
    propagator = WavePropagator()
    
    # Create a Gaussian beam
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2)/2) * np.sin(5*X)
    
    # Propagate
    propagated = propagator.propagate(
        field,
        wavelength=500e-9,
        distance=0.1,
        pixel_size=5e-6
    )
    
    # Energy should be conserved
    assert np.allclose(
        np.sum(np.abs(field)**2),
        np.sum(np.abs(propagated)**2),
        rtol=1e-3
    )

def test_propagator_zero_input():
    """Test propagation with zero input field"""
    propagator = WavePropagator()
    field = np.zeros((256, 256))
    
    propagated = propagator.propagate(
        field,
        wavelength=500e-9,
        distance=0.1,
        pixel_size=5e-6
    )
    
    assert np.allclose(propagated, 0)

def test_propagator_nan_handling():
    """Test propagation with NaN input"""
    propagator = WavePropagator()
    field = np.zeros((256, 256))
    field[0, 0] = np.nan
    
    with pytest.raises(ValueError):
        propagator.propagate(
            field,
            wavelength=500e-9,
            distance=0.1,
            pixel_size=5e-6
        )

def test_propagator_large_distance():
    """Test propagation with large distances"""
    propagator = WavePropagator()
    
    # Create a Gaussian beam
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2)/2)
    
    # Test multiple distances
    distances = [0.1, 1.0, 10.0]
    for distance in distances:
        propagated = propagator.propagate(
            field,
            wavelength=500e-9,
            distance=distance,
            pixel_size=5e-6
        )
        assert np.all(np.isfinite(propagated))
        assert propagated.shape == field.shape

def test_propagator_wavelength_limits():
    """Test propagation with different wavelengths"""
    propagator = WavePropagator()
    
    # Create a Gaussian beam
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2)/2)
    
    # Test wavelength limits
    wavelengths = [200e-9, 500e-9, 1000e-9, 2000e-9]
    for wavelength in wavelengths:
        propagated = propagator.propagate(
            field,
            wavelength=wavelength,
            distance=0.1,
            pixel_size=5e-6
        )
        assert np.all(np.isfinite(propagated))
        assert propagated.shape == field.shape

def test_propagator_pixel_size_limits():
    """Test propagation with different pixel sizes"""
    propagator = WavePropagator()
    
    # Create a Gaussian beam
    x = np.linspace(-10, 10, 256)
    y = np.linspace(-10, 10, 256)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2)/2)
    
    # Test pixel size limits
    pixel_sizes = [1e-6, 5e-6, 10e-6, 20e-6]
    for pixel_size in pixel_sizes:
        propagated = propagator.propagate(
            field,
            wavelength=500e-9,
            distance=0.1,
            pixel_size=pixel_size
        )
        assert np.all(np.isfinite(propagated))
        assert propagated.shape == field.shape

def test_propagator_field_size():
    """Test propagation with different field sizes"""
    propagator = WavePropagator()
    
    # Test different field sizes
    sizes = [64, 128, 256, 512]
    for size in sizes:
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        field = np.exp(-(X**2 + Y**2)/2)
        
        propagated = propagator.propagate(
            field,
            wavelength=500e-9,
            distance=0.1,
            pixel_size=5e-6
        )
        assert propagated.shape == field.shape 