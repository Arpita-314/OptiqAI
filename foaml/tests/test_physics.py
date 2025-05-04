from physics.components import Grating
from physics.propagation import angular_spectrum
import numpy as np

def test_grating_field():
    grating = Grating(period=1e-6)
    x = np.linspace(0, 10e-6, 1000)
    field = grating.compute_field(x, wavelength=633e-9)
    assert np.allclose(np.abs(field), 1.0)  # Phase-only grating

def test_propagation():
    field = np.ones(1000, dtype=complex)  # Plane wave
    propagated = angular_spectrum(field, wavelength=633e-9, distance=1e-3, dx=1e-6)
    assert np.allclose(np.sum(np.abs(propagated)**2), 1000)  # Energy conservation