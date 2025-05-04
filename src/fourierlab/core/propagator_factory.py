import numpy as np
import torch
import cupy as cp
from typing import Optional, Union, Dict, Any
from .propagator import WavePropagator
from .gpu_propagator import GPUWavePropagator

class PropagatorFactory:
    """Factory class for creating wave propagators with appropriate feature sets."""
    
    @staticmethod
    def create_propagator(
        mode: str = 'auto',
        device: Optional[int] = None,
        **kwargs
    ) -> Union[WavePropagator, GPUWavePropagator]:
        """
        Create a wave propagator with appropriate feature set.
        
        Args:
            mode: 'auto', 'gpu', or 'cpu'
            device: GPU device number (for GPU mode)
            **kwargs: Additional arguments for propagator initialization
            
        Returns:
            Wave propagator instance
        """
        if mode == 'auto':
            # Check if CUDA is available
            try:
                if torch.cuda.is_available():
                    return GPUWavePropagator(device=device)
                else:
                    return WavePropagator()
            except Exception:
                return WavePropagator()
        elif mode == 'gpu':
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested but CUDA is not available")
            return GPUWavePropagator(device=device)
        elif mode == 'cpu':
            return WavePropagator()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'auto', 'gpu', or 'cpu'")
    
    @staticmethod
    def get_available_features(mode: str = 'auto') -> Dict[str, Any]:
        """
        Get available features for the specified mode.
        
        Args:
            mode: 'auto', 'gpu', or 'cpu'
            
        Returns:
            Dictionary of available features
        """
        if mode == 'auto':
            mode = 'gpu' if torch.cuda.is_available() else 'cpu'
            
        if mode == 'gpu':
            return {
                'max_field_size': 8192,  # Maximum field size for GPU
                'batch_processing': True,  # Support for batch processing
                'mixed_precision': True,  # Support for mixed precision
                'multi_gpu': True,  # Support for multiple GPUs
                'advanced_features': True,  # Advanced features like real-time visualization
                'optimization_level': 'high'  # High optimization level
            }
        else:  # CPU mode
            return {
                'max_field_size': 2048,  # Maximum field size for CPU
                'batch_processing': False,  # No batch processing
                'mixed_precision': False,  # No mixed precision
                'multi_gpu': False,  # No multi-GPU support
                'advanced_features': False,  # Basic features only
                'optimization_level': 'medium'  # Medium optimization level
            }
    
    @staticmethod
    def get_recommended_settings(mode: str = 'auto') -> Dict[str, Any]:
        """
        Get recommended settings for the specified mode.
        
        Args:
            mode: 'auto', 'gpu', or 'cpu'
            
        Returns:
            Dictionary of recommended settings
        """
        if mode == 'auto':
            mode = 'gpu' if torch.cuda.is_available() else 'cpu'
            
        if mode == 'gpu':
            return {
                'field_size': 2048,  # Recommended field size
                'batch_size': 32,  # Recommended batch size
                'precision': 'float32',  # Recommended precision
                'memory_limit': 0.8,  # Use 80% of available GPU memory
                'optimization_level': 'high'
            }
        else:  # CPU mode
            return {
                'field_size': 1024,  # Recommended field size
                'batch_size': 1,  # No batching
                'precision': 'float64',  # Higher precision for CPU
                'memory_limit': 0.5,  # Use 50% of available RAM
                'optimization_level': 'medium'
            } 