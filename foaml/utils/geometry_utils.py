import numpy as np

def make_grid(size: float, num_points: int) -> np.ndarray:
    """Create a 1D coordinate grid."""
    return np.linspace(-size/2, size/2, num_points)