import numpy as np

class WavePropagator:
    """Base class for wave propagation."""
    
    def __init__(self):
        """Initialize wave propagator."""
        pass
        
    def angular_spectrum(self, field, wavelength, distance):
        """Propagate field using angular spectrum method.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            
        Returns:
            Propagated field (numpy array)
        """
        # Compute FFT
        fft = np.fft.fft2(field)
        
        # Generate frequency grid
        kx = np.fft.fftfreq(field.shape[0])
        ky = np.fft.fftfreq(field.shape[1])
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        # Compute transfer function
        k = 2 * np.pi / wavelength
        k_squared = k**2 - (2*np.pi*kx_grid)**2 - (2*np.pi*ky_grid)**2
        mask = k_squared > 0
        transfer = np.zeros_like(k_squared, dtype=np.complex128)
        transfer[mask] = np.exp(1j * distance * np.sqrt(k_squared[mask]))
        
        # Apply and inverse FFT
        return np.fft.ifft2(fft * transfer)
        
    def rayleigh_sommerfeld(self, field, wavelength, distance):
        """Propagate field using Rayleigh-Sommerfeld diffraction.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            
        Returns:
            Propagated field (numpy array)
        """
        # Compute FFT
        fft = np.fft.fft2(field)
        
        # Generate frequency grid
        kx = np.fft.fftfreq(field.shape[0])
        ky = np.fft.fftfreq(field.shape[1])
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        # Compute transfer function
        k = 2 * np.pi / wavelength
        k_squared = k**2 - (2*np.pi*kx_grid)**2 - (2*np.pi*ky_grid)**2
        mask = k_squared > 0
        transfer = np.zeros_like(k_squared, dtype=np.complex128)
        transfer[mask] = np.exp(1j * distance * np.sqrt(k_squared[mask]))
        
        # Apply and inverse FFT
        return np.fft.ifft2(fft * transfer)
        
    def fresnel(self, field, wavelength, distance, dx, dy):
        """Propagate field using Fresnel diffraction.
        
        Args:
            field: Complex field (numpy array)
            wavelength: Wavelength in meters
            distance: Propagation distance in meters
            dx: Pixel size in x direction (meters)
            dy: Pixel size in y direction (meters)
            
        Returns:
            Propagated field (numpy array)
        """
        # Generate spatial frequency grid
        nx, ny = field.shape
        kx = np.fft.fftfreq(nx) / dx
        ky = np.fft.fftfreq(ny) / dy
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        # Compute Fresnel transfer function
        k = 2 * np.pi / wavelength
        transfer = np.exp(1j * distance * k * (1 - (wavelength * kx_grid)**2 - (wavelength * ky_grid)**2))
        
        # Apply FFT, transfer function, and inverse FFT
        fft = np.fft.fft2(field)
        return np.fft.ifft2(fft * transfer) 