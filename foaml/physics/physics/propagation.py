import numpy as np

def angular_spectrum(field: np.ndarray, wavelength: float, distance: float, dx: float) -> np.ndarray:
    """FFT-based propagation using the Angular Spectrum method."""
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(field.shape[0], dx)
    kernel = np.exp(1j * distance * np.sqrt(k**2 - (2*np.pi*fx)**2))
    return np.fft.ifft(np.fft.fft(field) * kernel)

def fresnel(field: np.ndarray, wavelength: float, distance: float, dx: float) -> np.ndarray:
    """Fresnel approximation for near-field diffraction."""
    # Your implementation here
    ...