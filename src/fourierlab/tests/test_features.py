import os
import torch
import numpy as np
from ..core.pattern_generator import PatternGenerator
from ..models.phase_retrieval import PhaseCNN
from ..utils.visualization import TrainingVisualizer
from ..physics.propagator import WavePropagator

def test_pattern_generation():
    """Test pattern generation with different types"""
    print("\nTesting Pattern Generation...")
    generator = PatternGenerator()
    size = 256
    wavelength = 632.8e-9
    pixel_size = 5e-6
    
    # Test different pattern types
    pattern_types = [
        'zone_plate',
        'vortex',
        'bessel',
        'hermite',
        'laguerre'
    ]
    
    for pattern_type in pattern_types:
        print(f"\nGenerating {pattern_type} pattern...")
        pattern = generator.generate_pattern(
            pattern_type=pattern_type,
            size=size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            order=2 if pattern_type == 'vortex' else None,
            n=1 if pattern_type == 'hermite' else None,
            m=1 if pattern_type == 'hermite' else None,
            p=0 if pattern_type == 'laguerre' else None,
            l=1 if pattern_type == 'laguerre' else None
        )
        
        # Verify pattern properties
        assert pattern.shape == (size, size), f"Wrong shape for {pattern_type}"
        assert np.all(np.isfinite(pattern)), f"Non-finite values in {pattern_type}"
        assert pattern.min() >= 0 and pattern.max() <= 1, f"Values out of range for {pattern_type}"
        print(f"✓ {pattern_type} pattern generated successfully")

def test_phase_retrieval():
    """Test phase retrieval with physical constraints"""
    print("\nTesting Phase Retrieval...")
    
    # Create model
    model = PhaseCNN(
        input_size=(256, 256),
        n_channels=1,
        n_filters=32,
        n_layers=3
    )
    
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
    
    # Test phase prediction
    print("Testing phase prediction...")
    phase = model.predict_phase(
        intensity,
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    
    # Verify phase properties
    assert phase.shape == (1, 1, 256, 256), "Wrong phase shape"
    assert torch.all(torch.isfinite(phase)), "Non-finite values in phase"
    assert torch.all(phase >= -np.pi) and torch.all(phase <= np.pi), "Phase values out of range"
    print("✓ Phase prediction successful")

def test_visualization():
    """Test visualization tools"""
    print("\nTesting Visualization Tools...")
    
    # Create visualizer
    visualizer = TrainingVisualizer(
        log_dir='test_logs',
        save_dir='test_visualizations'
    )
    
    # Generate test data
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
    
    # Create model and get prediction
    model = PhaseCNN()
    phase = model.predict_phase(intensity)
    
    # Test visualization methods
    print("Testing visualization methods...")
    
    # Test metrics logging
    visualizer.log_metrics({'loss': 0.1, 'accuracy': 0.9}, step=0)
    print("✓ Metrics logging successful")
    
    # Test phase prediction visualization
    visualizer.log_phase_prediction(intensity, phase, step=0)
    print("✓ Phase prediction visualization successful")
    
    # Test optimization history
    history = [{'loss': 0.1}, {'loss': 0.05}, {'loss': 0.01}]
    visualizer.log_optimization_history(history)
    print("✓ Optimization history visualization successful")
    
    # Test model architecture logging
    visualizer.log_model_architecture(model, (1, 256, 256))
    print("✓ Model architecture logging successful")
    
    # Clean up
    visualizer.close()
    print("✓ Visualization tests completed")

def test_evaluation():
    """Test evaluation metrics"""
    print("\nTesting Evaluation Metrics...")
    
    # Create test data
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
    
    # Create model and get prediction
    model = PhaseCNN()
    predicted_phase = model.predict_phase(intensity)
    
    # Calculate intensity from predicted phase
    propagator = WavePropagator()
    field = torch.exp(1j * predicted_phase)
    predicted_intensity = propagator.calculate_intensity(field)
    predicted_intensity = predicted_intensity / predicted_intensity.max()
    
    # Calculate metrics
    from ..evaluation.evaluate import calculate_ssim
    
    # Test MSE
    mse = torch.mean((predicted_intensity - intensity)**2)
    print(f"✓ MSE: {mse.item():.6f}")
    
    # Test PSNR
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"✓ PSNR: {psnr.item():.6f}")
    
    # Test SSIM
    ssim = calculate_ssim(predicted_intensity, intensity)
    print(f"✓ SSIM: {ssim.item():.6f}")
    
    print("✓ Evaluation metrics calculated successfully")

def main():
    """Run all tests"""
    print("Starting feature tests...")
    
    # Create test directories
    os.makedirs('test_logs', exist_ok=True)
    os.makedirs('test_visualizations', exist_ok=True)
    
    # Run tests
    test_pattern_generation()
    test_phase_retrieval()
    test_visualization()
    test_evaluation()
    
    print("\nAll tests completed successfully!")
    
    # Clean up
    import shutil
    shutil.rmtree('test_logs')
    shutil.rmtree('test_visualizations')

if __name__ == '__main__':
    main() 