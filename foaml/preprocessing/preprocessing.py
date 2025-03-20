import numpy as np
import torch
import torch.fft as fft
from scipy import signal

class FourierPreprocessing:
    def __init__(self, pixel_size, wavelength):
        self.pixel_size = pixel_size  # in μm
        self.wavelength = wavelength  # in nm
        self.wavelength_m = wavelength * 1e-9  # Convert to meters

    def dc_removal(self, data):
        """Remove DC component (mean) from the data."""
        return data - torch.mean(data)

    def apply_window(self, data, window_type='tukey', alpha=0.25):
        """Apply windowing to reduce edge effects."""
        if window_type == 'tukey':
            window = torch.from_numpy(signal.tukey(data.shape[-1], alpha=alpha)).to(data.device)
        elif window_type == 'hamming':
            window = torch.from_numpy(np.hamming(data.shape[-1])).to(data.device)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
        
        if data.dim() == 2:
            window = window.unsqueeze(0) * window.unsqueeze(1)
        return data * window

    def fft2(self, data):
        """Perform 2D FFT."""
        return fft.fftshift(fft.fft2(fft.ifftshift(data)))

    def ifft2(self, data):
        """Perform 2D inverse FFT."""
        return fft.fftshift(fft.ifft2(fft.ifftshift(data)))

    def calculate_spatial_frequencies(self, shape):
        """Calculate spatial frequencies for given data shape."""
        ny, nx = shape
        dx = self.pixel_size * 1e-6  # Convert to meters
        fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
        return np.meshgrid(fx, fy)

    def phase_unwrap(self, wrapped_phase):
        """Simple 2D phase unwrapping."""
        unwrapped_phase = np.unwrap(np.unwrap(wrapped_phase, axis=0), axis=1)
        return torch.from_numpy(unwrapped_phase).to(wrapped_phase.device)

    def preprocess(self, data, remove_dc=True, window_type='tukey', unwrap_phase=False):
        """Main preprocessing pipeline."""
        if remove_dc:
            data = self.dc_removal(data)
        
        data = self.apply_window(data, window_type)
        
        if data.dtype == torch.complex64 or data.dtype == torch.complex128:
            amplitude = torch.abs(data)
            phase = torch.angle(data)
            if unwrap_phase:
                phase = self.phase_unwrap(phase)
            return amplitude, phase
        else:
            return data

# Usage example
if __name__ == "__main__":
    # Simulating some sample data
    sample_data = torch.randn(256, 256) + 1j * torch.randn(256, 256)
    
    preprocessor = FourierPreprocessing(pixel_size=5, wavelength=632.8)
    amplitude, phase = preprocessor.preprocess(sample_data, remove_dc=True, window_type='tukey', unwrap_phase=True)
    
    print("Preprocessing complete.")
    print(f"Amplitude shape: {amplitude.shape}")
    print(f"Phase shape: {phase.shape}")
