import torch
import numpy as np
import time
from pathlib import Path
from ..physics.propagator import WavePropagator
from ..models.phase_retrieval import PhaseCNN
from ..core.pattern_generator import PatternGenerator

def benchmark_propagation():
    """Benchmark wave propagation on CPU vs GPU"""
    print("\nBenchmarking Wave Propagation...")
    
    # Initialize propagator
    propagator = WavePropagator()
    
    # Generate test field
    generator = PatternGenerator()
    pattern = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    field = np.exp(1j * pattern)
    
    # Convert to tensor
    field_tensor = torch.from_numpy(field).float()
    
    # CPU propagation
    start_time = time.time()
    propagated_cpu = propagator.propagate(
        field,
        wavelength=632.8e-9,
        distance=0.1,
        pixel_size=5e-6
    )
    cpu_time = time.time() - start_time
    
    # GPU propagation (if available)
    if torch.cuda.is_available():
        field_tensor = field_tensor.cuda()
        start_time = time.time()
        propagated_gpu = propagator.propagate(
            field_tensor.cpu().numpy(),
            wavelength=632.8e-9,
            distance=0.1,
            pixel_size=5e-6
        )
        gpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.4f}s")
        print(f"GPU Time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU not available for propagation benchmark")

def benchmark_model_inference():
    """Benchmark model inference on CPU vs GPU"""
    print("\nBenchmarking Model Inference...")
    
    # Initialize model
    model = PhaseCNN()
    
    # Generate test pattern
    generator = PatternGenerator()
    pattern = generator.generate_pattern(
        pattern_type='vortex',
        size=256,
        wavelength=632.8e-9,
        pixel_size=5e-6,
        order=2
    )
    
    # Convert to tensor
    input_tensor = torch.from_numpy(pattern).float().unsqueeze(0).unsqueeze(0)
    
    # CPU inference
    start_time = time.time()
    output_cpu = model.predict_phase(
        input_tensor,
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    cpu_time = time.time() - start_time
    
    # GPU inference (if available)
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
        # Warm up
        for _ in range(10):
            _ = model.predict_phase(
                input_tensor,
                wavelength=632.8e-9,
                pixel_size=5e-6
            )
        
        # Benchmark
        start_time = time.time()
        output_gpu = model.predict_phase(
            input_tensor,
            wavelength=632.8e-9,
            pixel_size=5e-6
        )
        gpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.4f}s")
        print(f"GPU Time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Check memory usage
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("GPU not available for model inference benchmark")

def benchmark_batch_processing():
    """Benchmark batch processing on CPU vs GPU"""
    print("\nBenchmarking Batch Processing...")
    
    # Initialize components
    generator = PatternGenerator()
    model = PhaseCNN()
    
    # Generate batch of patterns
    batch_size = 16
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
    
    # CPU processing
    start_time = time.time()
    outputs_cpu = model.predict_phase(batch)
    cpu_time = time.time() - start_time
    
    # GPU processing (if available)
    if torch.cuda.is_available():
        model = model.cuda()
        batch = batch.cuda()
        
        # Warm up
        for _ in range(5):
            _ = model.predict_phase(batch)
        
        # Benchmark
        start_time = time.time()
        outputs_gpu = model.predict_phase(batch)
        gpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.4f}s")
        print(f"GPU Time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Check memory usage
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("GPU not available for batch processing benchmark")

def main():
    """Run all benchmarks"""
    print("Starting GPU Benchmarks...")
    
    # Create benchmark directory
    benchmark_dir = Path("benchmark_results")
    benchmark_dir.mkdir(exist_ok=True)
    
    # Run benchmarks
    benchmark_propagation()
    benchmark_model_inference()
    benchmark_batch_processing()
    
    print("\nBenchmarks completed!")

if __name__ == "__main__":
    main() 