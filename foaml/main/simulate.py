from physics.components import Grating
from physics.propagation import angular_spectrum
from utils.fft_utils import pad_field
import numpy as np
import matplotlib.pyplot as plt

# Define grating and grid
grating = Grating(period=1e-6, height=100e-9)
x = np.linspace(-5e-6, 5e-6, 1000)  # 10µm grid
field = grating.compute_field(x, wavelength=633e-9)

# Propagate
field_padded = pad_field(field)
propagated = angular_spectrum(field_padded, wavelength=633e-9, distance=1e-3, dx=x[1]-x[0])

# Plot
plt.plot(x, np.abs(propagated)**2)
plt.xlabel("Position (m)")
plt.ylabel("Intensity")
plt.show()