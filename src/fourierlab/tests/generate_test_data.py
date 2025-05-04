import numpy as np
from pathlib import Path
from skimage.io import imsave
from ..core.pattern_generator import PatternGenerator

def create_diffraction_pattern(size=256, wavelength=632.8e-9, pixel_size=5e-6):
    """Create a diffraction pattern with realistic noise"""
    # Create coordinate grid
    x = np.linspace(-size//2, size//2-1, size)
    y = np.linspace(-size//2, size//2-1, size)
    X, Y = np.meshgrid(x, y)
    
    # Generate pattern
    R = np.sqrt(X**2 + Y**2)
    pattern = np.sin(np.pi * R**2 / (wavelength * pixel_size))
    
    # Add Poisson noise to simulate experimental data
    pattern = np.abs(pattern)**2  # Convert to intensity
    pattern = pattern / pattern.max()  # Normalize
    pattern = np.random.poisson(pattern * 1000) / 1000  # Add Poisson noise
    
    return pattern

def create_test_dataset(output_dir="tests/data", n_samples=100):
    """Create a test dataset with various patterns"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pattern generator
    generator = PatternGenerator()
    
    # Pattern types and parameters
    pattern_configs = [
        {'type': 'vortex', 'order': 2},
        {'type': 'bessel', 'frequency': 10.0},
        {'type': 'hermite', 'n': 1, 'm': 1},
        {'type': 'laguerre', 'p': 0, 'l': 1},
        {'type': 'zone_plate', 'frequency': 10.0}
    ]
    
    # Generate patterns
    for i in range(n_samples):
        # Select random pattern type
        config = pattern_configs[i % len(pattern_configs)]
        
        # Generate pattern
        pattern = generator.generate_pattern(
            pattern_type=config['type'],
            size=256,
            wavelength=632.8e-9,
            pixel_size=5e-6,
            **{k: v for k, v in config.items() if k != 'type'}
        )
        
        # Add noise
        pattern = pattern + np.random.normal(0, 0.01, pattern.shape)
        pattern = np.clip(pattern, 0, 1)
        
        # Save pattern
        pattern_path = output_dir / f"pattern_{i:04d}.png"
        imsave(pattern_path, (pattern * 255).astype(np.uint8))
        
        # Generate corresponding diffraction pattern
        diff_pattern = create_diffraction_pattern()
        diff_path = output_dir / f"diffraction_{i:04d}.png"
        imsave(diff_path, (diff_pattern * 255).astype(np.uint8))

def create_benchmark_dataset(output_dir="benchmarks/data", n_samples=1000):
    """Create a larger dataset for benchmarking"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pattern generator
    generator = PatternGenerator()
    
    # Generate patterns
    for i in range(n_samples):
        # Generate random pattern
        pattern = generator.generate_pattern(
            pattern_type='vortex',
            size=256,
            wavelength=632.8e-9,
            pixel_size=5e-6,
            order=np.random.randint(1, 5)
        )
        
        # Add noise
        pattern = pattern + np.random.normal(0, 0.01, pattern.shape)
        pattern = np.clip(pattern, 0, 1)
        
        # Save pattern
        pattern_path = output_dir / f"pattern_{i:04d}.png"
        imsave(pattern_path, (pattern * 255).astype(np.uint8))

def main():
    """Generate all test data"""
    print("Generating test data...")
    
    # Create test dataset
    create_test_dataset()
    
    # Create benchmark dataset
    create_benchmark_dataset()
    
    print("Test data generation completed!")

if __name__ == "__main__":
    main() 