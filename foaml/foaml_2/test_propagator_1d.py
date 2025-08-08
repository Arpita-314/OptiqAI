import numpy as np
from src.propagator_1d import fourier_propagate_1d

def test_propagation_conserves_energy():
    field = np.random.rand(1024)
    propagated = fourier_propagate_1d(field, 500e-9, 1e-3)
    assert np.allclose(np.sum(np.abs(field)**2), np.sum(np.abs(propagated)**2), rtol=1e-3)