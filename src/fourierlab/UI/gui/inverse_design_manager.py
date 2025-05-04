import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image

from fourierlab.core.wave_propagator import WavePropagator
from fourierlab.core.phase_mask import PhaseMaskGenerator
from fourierlab.core.pattern_generator import PatternGenerator
from fourierlab.core.optical_simulator import OpticalSimulator

class InverseDesignManager(QObject):
    """Manager for inverse design with physics-based simulation."""
    
    progress_updated = pyqtSignal(int, dict)  # iteration, metrics
    generation_complete = pyqtSignal(torch.Tensor, torch.Tensor)  # phase_mask, simulated_output
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.phase_generator = None
        self.wave_propagator = None
        self.pattern_generator = PatternGenerator()
        self.optical_simulator = OpticalSimulator()
    
    def set_parameters(self, **params):
        """Set simulation parameters."""
        self.params = params
        
        # Initialize wave propagator
        self.wave_propagator = WavePropagator(
            grid_size=(256, 256),  # Default size
            wavelength=params.get('wavelength', 632.8e-9),
            pixel_size=params.get('pixel_size', 5e-6),
            device='cuda'
        )
        
        # Initialize phase generator
        self.phase_generator = PhaseMaskGenerator()
        self.phase_generator.set_parameters(**params)
    
    def load_target(self, target_type, image_path=None, size=None, pattern_type=None, pattern_params=None):
        """Load or generate target pattern/image"""
        try:
            if target_type == "Image":
                if image_path is None:
                    raise ValueError("Image path is required for Image target type")
                return self._load_image(image_path, size)
            else:  # Pattern
                if pattern_type is None:
                    raise ValueError("Pattern type is required for Pattern target type")
                return self._generate_pattern(pattern_type, size, pattern_params)
        except Exception as e:
            self.error_occurred.emit(str(e))
            return None
    
    def _load_image(self, image_path, size):
        """Load and preprocess image"""
        try:
            # Load image
            image = np.array(Image.open(image_path).convert('L'))
            
            # Resize if needed
            if size is not None:
                image = np.array(Image.fromarray(image).resize(size))
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return torch.from_numpy(image)
        except Exception as e:
            self.error_occurred.emit(f"Failed to load image: {str(e)}")
            return None
    
    def _generate_pattern(self, pattern_type, size, params):
        """Generate pattern"""
        try:
            pattern = self.pattern_generator.generate_pattern(
                pattern_type=pattern_type,
                size=size[0],  # Assuming square pattern
                width=params.get('width', 25),
                frequency=params.get('frequency', 10)
            )
            return torch.from_numpy(pattern)
        except Exception as e:
            self.error_occurred.emit(f"Failed to generate pattern: {str(e)}")
            return None
    
    def generate_phase_mask(self, target, wavelength, pixel_size, iterations):
        """Generate phase mask using physics-based optimization."""
        try:
            # Convert target to tensor if needed
            if not isinstance(target, torch.Tensor):
                target = torch.from_numpy(target).to('cuda')
            
            # Initialize phase mask
            phase_mask = torch.zeros_like(target, device='cuda')
            phase_mask.requires_grad = True
            
            # Setup optimizer
            optimizer = torch.optim.Adam([phase_mask], lr=self.params.get('learning_rate', 0.01))
            
            # Optimization loop
            for i in range(iterations):
                optimizer.zero_grad()
                
                # Compute complex field
                field = torch.exp(1j * phase_mask)
                
                # Propagate with physics simulation
                propagated = self.wave_propagator.propagate(
                    field,
                    distance=self.params.get('propagation_distance', 0.1),
                    method='angular_spectrum',
                    medium=self.params.get('medium'),
                    scatterers=self.params.get('scatterers')
                )
                
                # Compute intensity
                intensity = self.wave_propagator.get_intensity(propagated)
                
                # Compute loss
                loss = torch.mean((intensity - target)**2)
                
                # Add regularization
                if self.params.get('smoothness_weight', 0) > 0:
                    smoothness = torch.mean(torch.abs(phase_mask[1:] - phase_mask[:-1])) + \
                               torch.mean(torch.abs(phase_mask[:, 1:] - phase_mask[:, :-1]))
                    loss += self.params['smoothness_weight'] * smoothness
                
                # Backpropagate
                loss.backward()
                optimizer.step()
                
                # Enforce phase constraints
                with torch.no_grad():
                    phase_mask.data.clamp_(0, 2*np.pi)
                
                # Emit progress
                metrics = {
                    'loss': loss.item(),
                    'smoothness': smoothness.item() if 'smoothness' in locals() else 0
                }
                self.progress_updated.emit(i + 1, metrics)
            
            # Final propagation
            with torch.no_grad():
                field = torch.exp(1j * phase_mask)
                final_output = self.wave_propagator.propagate(
                    field,
                    distance=self.params.get('propagation_distance', 0.1),
                    method='angular_spectrum',
                    medium=self.params.get('medium'),
                    scatterers=self.params.get('scatterers')
                )
                final_intensity = self.wave_propagator.get_intensity(final_output)
            
            # Emit completion
            self.generation_complete.emit(phase_mask, final_intensity)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            return None
    
    def save_results(self, phase_mask_path=None, output_path=None):
        """Save generated results"""
        try:
            if phase_mask_path:
                phase_mask = self.phase_generator.get_phase_mask()
                if phase_mask is not None:
                    phase_mask_np = phase_mask.detach().cpu().numpy()
                    Image.fromarray((phase_mask_np * 255).astype(np.uint8)).save(phase_mask_path)
            
            if output_path:
                output = self.phase_generator.get_output()
                if output is not None:
                    output_np = output.detach().cpu().numpy()
                    Image.fromarray((output_np * 255).astype(np.uint8)).save(output_path)
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to save results: {str(e)}")
            return False 