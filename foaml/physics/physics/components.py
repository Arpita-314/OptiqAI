import numpy as np

class Grating:
    def __init__(self, period: float, height: float):
        self.period = period  # meters
        self.height = height  # meters
    
    def compute_field(self, x: np.ndarray, wavelength: float) -> np.ndarray:
        """Generate a periodic phase profile."""
        return np.exp(1j * 2*np.pi * (x % self.period) / self.period)

class Lens:
    def __init__(self, focal_length: float):
        self.focal_length = focal_length
    
    def compute_field(self, x: np.ndarray, wavelength: float) -> np.ndarray:
        """Quadratic phase profile for a thin lens."""
        return np.exp(-1j * np.pi * x**2 / (wavelength * self.focal_length))