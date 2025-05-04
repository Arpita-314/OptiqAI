import os
import sys
import torch
import subprocess
from pathlib import Path

def check_cuda_installation():
    """Check CUDA installation and environment."""
    print("Checking CUDA installation...")
    
    # Check if CUDA is available in PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available in PyTorch: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"CUDA_PATH is set to: {cuda_path}")
    else:
        print("CUDA_PATH is not set!")
        
        # Try to find CUDA installation
        possible_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]
                    cuda_path = os.path.join(base_path, latest_version)
                    print(f"Found CUDA installation at: {cuda_path}")
                    print("Please set CUDA_PATH environment variable to this path")
                    break

def setup_cuda_environment():
    """Setup CUDA environment variables."""
    print("\nSetting up CUDA environment...")
    
    # Get CUDA path
    cuda_path = os.environ.get('CUDA_PATH')
    if not cuda_path:
        print("CUDA_PATH not found. Please install CUDA Toolkit first.")
        return False
    
    # Add CUDA paths to system PATH
    cuda_bin = os.path.join(cuda_path, 'bin')
    cuda_lib = os.path.join(cuda_path, 'libnvvp')
    
    if cuda_bin not in os.environ['PATH']:
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
    
    if cuda_lib not in os.environ['PATH']:
        os.environ['PATH'] = cuda_lib + os.pathsep + os.environ['PATH']
    
    print("CUDA environment setup complete!")
    return True

if __name__ == "__main__":
    check_cuda_installation()
    if setup_cuda_environment():
        print("\nCUDA setup complete! You can now run the application with GPU support.")
    else:
        print("\nPlease install CUDA Toolkit and try again.") 