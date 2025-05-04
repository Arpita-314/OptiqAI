import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple
import json

class PhaseRetrievalDataset(Dataset):
    """Dataset for phase retrieval training"""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing data
            transform: Optional transform to apply
            target_size: Target size for images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Get list of samples
        self.samples = self._get_samples()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_samples(self) -> list:
        """Get list of samples"""
        samples = []
        
        # Look for intensity and phase pairs
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_intensity.png'):
                # Get corresponding phase file
                phase_filename = filename.replace('_intensity.png', '_phase.png')
                phase_path = os.path.join(self.data_dir, phase_filename)
                
                if os.path.exists(phase_path):
                    # Get sample ID
                    sample_id = filename.replace('_intensity.png', '')
                    
                    # Get metadata
                    metadata = self.metadata.get(sample_id, {})
                    
                    samples.append({
                        'id': sample_id,
                        'intensity_path': os.path.join(self.data_dir, filename),
                        'phase_path': phase_path,
                        'wavelength': metadata.get('wavelength', 633e-9),  # Default to 633nm
                        'pixel_size': metadata.get('pixel_size', 1e-6)    # Default to 1μm
                    })
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - intensity: Intensity image tensor
                - phase: Phase image tensor
                - wavelength: Wavelength value
                - pixel_size: Pixel size value
        """
        sample = self.samples[idx]
        
        # Load intensity image
        intensity = Image.open(sample['intensity_path']).convert('L')
        intensity = intensity.resize(self.target_size)
        intensity = np.array(intensity, dtype=np.float32) / 255.0
        
        # Load phase image
        phase = Image.open(sample['phase_path']).convert('L')
        phase = phase.resize(self.target_size)
        phase = np.array(phase, dtype=np.float32) / 255.0
        phase = (phase * 2 - 1) * np.pi  # Scale to [-π, π]
        
        # Convert to tensors
        intensity = torch.from_numpy(intensity).unsqueeze(0)
        phase = torch.from_numpy(phase).unsqueeze(0)
        
        # Apply transform if specified
        if self.transform:
            intensity = self.transform(intensity)
            phase = self.transform(phase)
        
        return {
            'intensity': intensity,
            'phase': phase,
            'wavelength': torch.tensor(sample['wavelength']),
            'pixel_size': torch.tensor(sample['pixel_size'])
        }

def create_synthetic_dataset(
    output_dir: str,
    n_samples: int = 1000,
    image_size: Tuple[int, int] = (256, 256),
    wavelength_range: Tuple[float, float] = (400e-9, 800e-9),
    pixel_size_range: Tuple[float, float] = (0.5e-6, 2e-6)
):
    """
    Create synthetic dataset for phase retrieval
    
    Args:
        output_dir: Output directory
        n_samples: Number of samples to generate
        image_size: Image size
        wavelength_range: Range of wavelengths
        pixel_size_range: Range of pixel sizes
    """
    from ..physics.propagator import WavePropagator
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wave propagator
    propagator = WavePropagator()
    
    # Generate samples
    metadata = {}
    for i in range(n_samples):
        # Generate random phase
        phase = np.random.rand(*image_size) * 2 * np.pi - np.pi
        
        # Generate random wavelength and pixel size
        wavelength = np.random.uniform(*wavelength_range)
        pixel_size = np.random.uniform(*pixel_size_range)
        
        # Calculate intensity
        field = np.exp(1j * phase)
        intensity = propagator.calculate_intensity(field)
        
        # Normalize intensity
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        
        # Save images
        sample_id = f'sample_{i:04d}'
        
        # Save intensity
        intensity_img = Image.fromarray((intensity * 255).astype(np.uint8))
        intensity_img.save(os.path.join(output_dir, f'{sample_id}_intensity.png'))
        
        # Save phase
        phase_img = Image.fromarray(((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8))
        phase_img.save(os.path.join(output_dir, f'{sample_id}_phase.png'))
        
        # Save metadata
        metadata[sample_id] = {
            'wavelength': wavelength,
            'pixel_size': pixel_size
        }
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == '__main__':
    # Create synthetic dataset
    create_synthetic_dataset(
        output_dir='data/train',
        n_samples=1000
    )
    create_synthetic_dataset(
        output_dir='data/val',
        n_samples=100
    ) 