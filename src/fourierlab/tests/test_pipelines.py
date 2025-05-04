import os
import torch
import numpy as np
import pytest
from pathlib import Path
from ..core.pattern_generator import PatternGenerator
from ..models.phase_retrieval import PhaseCNN
from ..physics.propagator import WavePropagator
from ..utils.visualization import TrainingVisualizer

def test_simulate_analyze_export(tmp_path):
    """Test complete workflow: simulate → analyze → export"""
    # Create test directories
    data_dir = tmp_path / "test_data"
    output_dir = tmp_path / "test_output"
    data_dir.mkdir()
    output_dir.mkdir()
    
    # Initialize components
    generator = PatternGenerator()
    model = PhaseCNN()
    propagator = WavePropagator()
    visualizer = TrainingVisualizer(
        log_dir=str(output_dir / "logs"),
        save_dir=str(output_dir / "visualizations")
    )
    
    # 1. Generate test pattern
    pattern = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    
    # Save pattern
    pattern_path = data_dir / "test_pattern.npy"
    np.save(pattern_path, pattern)
    
    # 2. Simulate propagation
    field = np.exp(1j * pattern)  # Convert to complex field
    propagated = propagator.propagate(
        field,
        wavelength=632.8e-9,
        distance=0.1,
        pixel_size=5e-6
    )
    
    # Save propagated field
    propagated_path = data_dir / "propagated_field.npy"
    np.save(propagated_path, propagated)
    
    # 3. Analyze with model
    intensity = np.abs(propagated)**2
    intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)
    
    predicted_phase = model.predict_phase(
        intensity_tensor,
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    
    # 4. Export results
    # Save phase prediction
    phase_path = output_dir / "predicted_phase.npy"
    np.save(phase_path, predicted_phase.detach().cpu().numpy())
    
    # Log metrics
    metrics = {
        'mse': float(torch.mean((predicted_phase - torch.from_numpy(pattern).float())**2)),
        'mae': float(torch.mean(torch.abs(predicted_phase - torch.from_numpy(pattern).float())))
    }
    visualizer.log_metrics(metrics, step=0)
    
    # Verify outputs
    assert pattern_path.exists()
    assert propagated_path.exists()
    assert phase_path.exists()
    assert (output_dir / "logs").exists()
    assert (output_dir / "visualizations").exists()
    
    # Verify metrics
    assert metrics['mse'] < 1.0  # Reasonable error threshold
    assert metrics['mae'] < 1.0

def test_workflow_with_different_patterns(tmp_path):
    """Test workflow with different pattern types"""
    pattern_types = ['vortex', 'bessel', 'hermite', 'laguerre']
    
    for pattern_type in pattern_types:
        # Initialize components
        generator = PatternGenerator()
        model = PhaseCNN()
        propagator = WavePropagator()
        
        # Generate pattern
        pattern = generator.generate_pattern(
            pattern_type=pattern_type,
            size=256,
            wavelength=632.8e-9,
            pixel_size=5e-6,
            order=2 if pattern_type == 'vortex' else None,
            n=1 if pattern_type == 'hermite' else None,
            m=1 if pattern_type == 'hermite' else None,
            p=0 if pattern_type == 'laguerre' else None,
            l=1 if pattern_type == 'laguerre' else None
        )
        
        # Simulate and analyze
        field = np.exp(1j * pattern)
        propagated = propagator.propagate(
            field,
            wavelength=632.8e-9,
            distance=0.1,
            pixel_size=5e-6
        )
        
        intensity = np.abs(propagated)**2
        intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)
        
        predicted_phase = model.predict_phase(
            intensity_tensor,
            wavelength=632.8e-9,
            pixel_size=5e-6
        )
        
        # Verify outputs
        assert torch.all(torch.isfinite(predicted_phase))
        assert predicted_phase.shape == (1, 1, 256, 256)

def test_workflow_error_handling(tmp_path):
    """Test workflow error handling"""
    # Initialize components
    generator = PatternGenerator()
    model = PhaseCNN()
    propagator = WavePropagator()
    
    # Test invalid pattern type
    with pytest.raises(ValueError):
        generator.generate_pattern(
            pattern_type='invalid_pattern',
            size=256,
            wavelength=632.8e-9,
            pixel_size=5e-6
        )
    
    # Test invalid wavelength
    with pytest.raises(ValueError):
        propagator.propagate(
            np.zeros((256, 256)),
            wavelength=-1.0,
            distance=0.1,
            pixel_size=5e-6
        )
    
    # Test invalid input to model
    with pytest.raises(ValueError):
        model.predict_phase(
            torch.full((1, 1, 256, 256), float('nan'))
        )

def test_workflow_performance(tmp_path):
    """Test workflow performance"""
    import time
    
    # Initialize components
    generator = PatternGenerator()
    model = PhaseCNN()
    propagator = WavePropagator()
    
    # Generate pattern
    start_time = time.time()
    pattern = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    generation_time = time.time() - start_time
    
    # Simulate
    start_time = time.time()
    field = np.exp(1j * pattern)
    propagated = propagator.propagate(
        field,
        wavelength=632.8e-9,
        distance=0.1,
        pixel_size=5e-6
    )
    simulation_time = time.time() - start_time
    
    # Analyze
    start_time = time.time()
    intensity = np.abs(propagated)**2
    intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)
    predicted_phase = model.predict_phase(
        intensity_tensor,
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    analysis_time = time.time() - start_time
    
    # Verify performance
    assert generation_time < 1.0  # Should be fast
    assert simulation_time < 1.0  # Should be fast
    assert analysis_time < 2.0  # Model inference might take longer 