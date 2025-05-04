import numpy as np
import pytest
from fourierlab.core.rust_bindings import WavePropagator, PatternGenerator, PhaseMaskOptimizer

def test_wave_propagator():
    """Test wave propagation methods."""
    # Initialize propagator
    propagator = WavePropagator(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Create test field (Gaussian beam)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    field = np.exp(-(X**2 + Y**2) / 0.1)
    
    # Test single field propagation
    distance = 0.1
    result_as = propagator.angular_spectrum(field, distance)
    result_rs = propagator.rayleigh_sommerfeld(field, distance)
    result_fr = propagator.fresnel(field, distance)
    
    assert result_as.shape == field.shape
    assert result_rs.shape == field.shape
    assert result_fr.shape == field.shape
    assert np.all(np.isfinite(result_as))
    assert np.all(np.isfinite(result_rs))
    assert np.all(np.isfinite(result_fr))
    
    # Test batch propagation
    batch_size = 4
    fields = np.stack([field] * batch_size)
    result_as_batch = propagator.angular_spectrum_batch(fields, distance)
    result_rs_batch = propagator.rayleigh_sommerfeld_batch(fields, distance)
    result_fr_batch = propagator.fresnel_batch(fields, distance)
    
    assert result_as_batch.shape == field.shape
    assert result_rs_batch.shape == field.shape
    assert result_fr_batch.shape == field.shape
    assert np.all(np.isfinite(result_as_batch))
    assert np.all(np.isfinite(result_rs_batch))
    assert np.all(np.isfinite(result_fr_batch))
    
    # Test error handling
    with pytest.raises(ValueError):
        propagator.angular_spectrum(field, -1.0)
    with pytest.raises(ValueError):
        propagator.angular_spectrum_batch(fields, -1.0)
    with pytest.raises(ValueError):
        propagator.angular_spectrum(np.array([]), distance)

def test_pattern_generator():
    """Test pattern generation methods."""
    # Initialize generator
    generator = PatternGenerator(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Test single pattern generation
    size = (64, 64)
    width = 0.1
    frequency = 10.0
    
    pattern_types = ["cross", "circle", "square", "grating", "spiral", "vortex", "bessel", "hermite", "laguerre"]
    for pattern_type in pattern_types:
        if pattern_type in ["vortex", "hermite", "laguerre"]:
            pattern = generator.generate_pattern(pattern_type, size, width, frequency, order=1)
        else:
            pattern = generator.generate_pattern(pattern_type, size, width, frequency)
        
        assert pattern.shape == size
        assert np.all(np.isfinite(pattern))
    
    # Test batch pattern generation
    batch_size = 4
    for pattern_type in pattern_types:
        if pattern_type in ["vortex", "hermite", "laguerre"]:
            pattern = generator.generate_pattern_batch(pattern_type, size, width, frequency, order=1, batch_size=batch_size)
        else:
            pattern = generator.generate_pattern_batch(pattern_type, size, width, frequency, batch_size=batch_size)
        
        assert pattern.shape == size
        assert np.all(np.isfinite(pattern))
    
    # Test error handling
    with pytest.raises(ValueError):
        generator.generate_pattern("unknown", size, width, frequency)
    with pytest.raises(ValueError):
        generator.generate_pattern("vortex", size, width, frequency)  # Missing order
    with pytest.raises(ValueError):
        generator.generate_pattern("cross", size, -1.0, frequency)
    with pytest.raises(ValueError):
        generator.generate_pattern("cross", size, width, -1.0)

def test_phase_mask_optimizer():
    """Test phase mask optimization methods."""
    # Initialize optimizer
    optimizer = PhaseMaskOptimizer(wavelength=632.8e-9, pixel_size=10e-6)
    
    # Create test target (Gaussian beam)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    target = np.exp(-(X**2 + Y**2) / 0.1)
    
    # Test single phase mask optimization
    distance = 0.1
    result = optimizer.optimize_phase_mask(target, distance=distance)
    
    assert result.shape == target.shape
    assert np.all(np.isfinite(result))
    assert np.all(np.abs(result) <= 1.0)  # Phase should be normalized
    
    # Test batch phase mask optimization
    batch_size = 4
    targets = np.stack([target] * batch_size)
    result_batch = optimizer.optimize_phase_mask_batch(targets, distance=distance)
    
    assert result_batch.shape == target.shape
    assert np.all(np.isfinite(result_batch))
    assert np.all(np.abs(result_batch) <= 1.0)
    
    # Test with initial phase
    initial_phase = np.random.rand(*target.shape)
    result_with_initial = optimizer.optimize_phase_mask(target, initial_phase, distance)
    
    assert result_with_initial.shape == target.shape
    assert np.all(np.isfinite(result_with_initial))
    assert np.all(np.abs(result_with_initial) <= 1.0)
    
    # Test error handling
    with pytest.raises(ValueError):
        optimizer.optimize_phase_mask(target, distance=-1.0)
    with pytest.raises(ValueError):
        optimizer.optimize_phase_mask_batch(targets, distance=-1.0)
    with pytest.raises(ValueError):
        optimizer.optimize_phase_mask(np.array([]), distance=distance)
    with pytest.raises(ValueError):
        optimizer.optimize_phase_mask(target, initial_phase=np.random.rand(32, 32), distance=distance)  # Size mismatch

if __name__ == "__main__":
    pytest.main([__file__]) 