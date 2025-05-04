import numpy as np
import torch
import torch.fft as fft

class OpticalSimulator:
    """Class for simulating optical field propagation"""
    
    def __init__(self):
        self.wavelength = 632.8e-9  # meters
        self.pixel_size = 5e-6  # meters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def simulate(self, phase_mask, wavelength=None, pixel_size=None):
        """Simulate optical propagation through phase mask"""
        # Update parameters if provided
        if wavelength is not None:
            self.wavelength = wavelength
        if pixel_size is not None:
            self.pixel_size = pixel_size
        
        # Convert phase mask to tensor if needed
        if isinstance(phase_mask, np.ndarray):
            phase_mask = torch.from_numpy(phase_mask).float()
        
        # Calculate propagation parameters
        dx = self.pixel_size
        k = 2 * np.pi / self.wavelength
        N = phase_mask.shape[0]
        
        # Create coordinate grids
        x = torch.linspace(-N//2, N//2-1, N) * dx
        y = torch.linspace(-N//2, N//2-1, N) * dx
        X, Y = torch.meshgrid(x, y)
        
        # Calculate transfer function
        H = torch.exp(1j * k * torch.sqrt(1 - (X**2 + Y**2) / (k**2 * dx**2)))
        H = torch.fft.fftshift(H)
        
        # Initial field
        field = torch.ones_like(phase_mask)
        
        # Apply phase mask
        phase = torch.exp(1j * phase_mask)
        field = field * phase
        
        # Propagate field
        field_fft = torch.fft.fft2(field)
        field_prop = torch.fft.ifft2(field_fft * H)
        
        # Calculate intensity
        intensity = torch.abs(field_prop)**2
        return intensity / intensity.max()
    
    def calculate_psf(self, wavelength=None, pixel_size=None):
        """Calculate point spread function"""
        # Create delta function
        N = 256  # Size of PSF
        psf = torch.zeros((N, N))
        psf[N//2, N//2] = 1
        
        # Simulate propagation
        return self.simulate(psf, wavelength, pixel_size)
    
    def calculate_mtf(self, wavelength=None, pixel_size=None):
        """Calculate modulation transfer function"""
        # Get PSF
        psf = self.calculate_psf(wavelength, pixel_size)
        
        # Calculate MTF
        psf_fft = torch.fft.fft2(psf)
        mtf = torch.abs(psf_fft)
        mtf = mtf / mtf.max()
        
        return mtf
    
    def propagate(self, field, distance):
        """Propagate optical field by specified distance"""
        if not all([self.wavelength, self.pixel_size]):
            raise ValueError("Simulator not properly set up")
        
        # Calculate spatial frequencies
        nx, ny = field.shape
        dx = self.pixel_size
        k = 2 * np.pi / self.wavelength
        
        # Create frequency grid
        fx = fft.fftfreq(nx, dx)
        fy = fft.fftfreq(ny, dx)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Calculate transfer function
        H = torch.exp(1j * k * distance * torch.sqrt(1 - (self.wavelength * FX) ** 2 - (self.wavelength * FY) ** 2))
        H = H.to(self.device)
        
        # Apply transfer function
        field_fft = fft.fft2(field)
        field_prop_fft = field_fft * H
        field_prop = fft.ifft2(field_prop_fft)
        
        return field_prop
    
    def calculate_intensity(self, field):
        """Calculate intensity from complex field"""
        return torch.abs(field) ** 2
    
    def calculate_phase(self, field):
        """Calculate phase from complex field"""
        return torch.angle(field) 