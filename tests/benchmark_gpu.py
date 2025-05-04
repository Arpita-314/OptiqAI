import time
import numpy as np
import torch
from fourierlab.core.propagator import WavePropagator

def generate_test_field(size=2048):
    """Generate a complex test field."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    field = np.exp(-r**2/0.1) * np.exp(1j * 2*np.pi * r)
    return field

def benchmark_propagation():
    """Benchmark CPU propagation."""
    # Test parameters
    size = 2048
    wavelength = 500e-9  # 500 nm
    distance = 0.1  # 10 cm
    
    # Generate test field
    field = generate_test_field(size)
    
    # Initialize propagator
    cpu_propagator = WavePropagator()
    
    # Benchmark different methods
    methods = ['angular_spectrum', 'rayleigh_sommerfeld', 'fresnel']
    
    print("\nPropagation Benchmark Results:")
    print(f"Field size: {size}x{size}")
    
    for method in methods:
        # Warm up
        if method == 'fresnel':
            _ = getattr(cpu_propagator, method)(field, wavelength, distance, 5e-6, 5e-6)
        else:
            _ = getattr(cpu_propagator, method)(field, wavelength, distance)
        
        # Benchmark
        start = time.time()
        if method == 'fresnel':
            _ = getattr(cpu_propagator, method)(field, wavelength, distance, 5e-6, 5e-6)
        else:
            _ = getattr(cpu_propagator, method)(field, wavelength, distance)
        cpu_time = time.time() - start
        
        print(f"\n{method.replace('_', ' ').title()} Method:")
        print(f"CPU time: {cpu_time:.2f}s")

if __name__ == "__main__":
    benchmark_propagation() 