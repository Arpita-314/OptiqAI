import numpy as np
import matplotlib.pyplot as plt

def fourier_propagate_1d(field, wavelength, distance):
    """
    Propagate a 1D wavefield using Fourier optics.
    :param field: Input wavefield (1D numpy array).
    :param wavelength: Light wavelength (meters).
    :param distance: Propagation distance (meters).
    :return: Propagated field.
    """
    # Step 1: FFT of input field
    fft_field = np.fft.fft(field)
    
    # Step 2: Frequency grid
    n = len(field)
    kx = np.fft.fftfreq(n)
    
    # Step 3: Transfer function
    transfer = np.exp(1j * 2 * np.pi * distance * np.sqrt((1/wavelength)**2 - kx**2))
    
    # Step 4: Inverse FFT
    output = np.fft.ifft(fft_field * transfer)
    return output

# Test with a simple Gaussian beam
if __name__ == "__main__":
    x = np.linspace(-10e-6, 10e-6, 1024)  # 10μm window
    field = np.exp(-x**2 / (1e-6)**2)      # Gaussian beam
    propagated = fourier_propagate_1d(field, 500e-9, 1e-3)  # 500nm light, 1mm distance
    
    plt.plot(np.abs(propagated)**2)
    plt.title("Propagated Beam Intensity")
    plt.savefig("propagation_1d.png")