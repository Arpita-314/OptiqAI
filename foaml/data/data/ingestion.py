import numpy as np
from scipy.io import loadmat
import torch

class DataIngestion:
    def __init__(self):
        self.data_types = ['complex_field', 'intensity', 'psf', 'wavefront']

    def load_data(self, file_path):
        """Load data from file and detect type."""
        # For this example, we'll assume .mat files
        data = loadmat(file_path)
        
        # Detect data type
        if 'complex_field' in data:
            return self.process_complex_field(data['complex_field'])
        elif 'intensity' in data:
            return self.process_intensity(data['intensity'])
        # Add more conditions for other data types
        else:
            raise ValueError("Unknown data type")

    def process_complex_field(self, data):
        """Process complex field data."""
        return torch.from_numpy(data).to(torch.complex64)

    def process_intensity(self, data):
        """Process intensity data."""
        return torch.from_numpy(data).to(torch.float32)

    def get_user_input(self):
        """Get user input for data parameters."""
        data_type = input("Enter data type (complex_field, intensity, psf, wavefront): ")
        pixel_size = float(input("Enter pixel size (in μm): "))
        wavelength = float(input("Enter wavelength (in nm): "))
        return data_type, pixel_size, wavelength

# Usage example
if __name__ == "__main__":
    ingestion = DataIngestion()
    data_type, pixel_size, wavelength = ingestion.get_user_input()
    print(f"Data type: {data_type}")
    print(f"Pixel size: {pixel_size} μm")
    print(f"Wavelength: {wavelength} nm")
