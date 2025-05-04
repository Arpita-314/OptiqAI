import numpy as np

def pad_field(field: np.ndarray, pad_width: int = 100) -> np.ndarray:
    """Zero-pad a field for better FFT accuracy."""
    return np.pad(field, pad_width, mode='constant')

def fftshift_2d(field: np.ndarray) -> np.ndarray:
    """Shift FFT output for visualization."""
    return np.fft.fftshift(field)