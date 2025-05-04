import torch
import numpy as np
import pytest
from ..models.phase_retrieval import PhaseCNN
from ..core.pattern_generator import PatternGenerator

def test_model_initialization():
    """Test model initialization with different parameters"""
    sizes = [(64, 64), (128, 128), (256, 256)]
    n_filters = [16, 32, 64]
    n_layers = [2, 3, 4]
    
    for size in sizes:
        for n_filter in n_filters:
            for n_layer in n_layers:
                model = PhaseCNN(
                    input_size=size,
                    n_filters=n_filter,
                    n_layers=n_layer
                )
                assert model.input_size == size
                assert model.n_filters == n_filter
                assert model.n_layers == n_layer

def test_model_forward_pass():
    """Test model forward pass with synthetic data"""
    model = PhaseCNN()
    
    # Generate synthetic intensity pattern
    generator = PatternGenerator()
    intensity = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    
    # Convert to tensor
    intensity = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    phase = model(intensity)
    
    # Check output properties
    assert phase.shape == (1, 1, 256, 256)
    assert torch.all(torch.isfinite(phase))
    assert torch.all(phase >= -np.pi) and torch.all(phase <= np.pi)

def test_model_phase_prediction():
    """Test phase prediction with physical constraints"""
    model = PhaseCNN()
    
    # Generate test pattern
    generator = PatternGenerator()
    intensity = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    
    # Convert to tensor
    intensity = torch.from_numpy(intensity).float().unsqueeze(0).unsqueeze(0)
    
    # Predict phase
    phase = model.predict_phase(
        intensity,
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    
    # Verify phase properties
    assert phase.shape == (1, 1, 256, 256)
    assert torch.all(torch.isfinite(phase))
    assert torch.all(phase >= -np.pi) and torch.all(phase <= np.pi)

def test_model_edge_cases():
    """Test model with edge cases"""
    model = PhaseCNN()
    
    # Test zero input
    zero_input = torch.zeros((1, 1, 256, 256))
    phase = model.predict_phase(zero_input)
    assert torch.all(torch.isfinite(phase))
    
    # Test uniform input
    uniform_input = torch.ones((1, 1, 256, 256))
    phase = model.predict_phase(uniform_input)
    assert torch.all(torch.isfinite(phase))
    
    # Test NaN input
    nan_input = torch.full((1, 1, 256, 256), float('nan'))
    with pytest.raises(ValueError):
        model.predict_phase(nan_input)

def test_model_save_load():
    """Test model saving and loading"""
    # Create and save model
    model = PhaseCNN()
    torch.save(model.state_dict(), 'test_model.pt')
    
    # Load model
    loaded_model = PhaseCNN()
    loaded_model.load_state_dict(torch.load('test_model.pt'))
    
    # Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)

def test_model_batch_processing():
    """Test model with batch processing"""
    model = PhaseCNN()
    
    # Generate batch of patterns
    generator = PatternGenerator()
    batch_size = 4
    patterns = []
    
    for _ in range(batch_size):
        pattern = generator.generate_pattern(
            pattern_type='vortex',
            size=256,
            wavelength=632.8e-9,
            pixel_size=5e-6,
            order=2
        )
        patterns.append(pattern)
    
    # Convert to tensor
    batch = torch.stack([
        torch.from_numpy(p).float().unsqueeze(0)
        for p in patterns
    ])
    
    # Process batch
    phases = model.predict_phase(batch)
    
    # Check output
    assert phases.shape == (batch_size, 1, 256, 256)
    assert torch.all(torch.isfinite(phases))
    assert torch.all(phases >= -np.pi) and torch.all(phases <= np.pi) 