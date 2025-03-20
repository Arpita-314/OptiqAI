import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any

class FourierBlock(nn.Module):
    """Custom Fourier layer with physics constraints"""
    def __init__(self, in_channels: int, out_channels: int, enforce_conservation: bool = True):
        super().__init__()
        self.enforce = enforce_conservation
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
    def forward(self, x: Tensor) -> Tensor:
        # Fourier domain processing
        x_fft = torch.fft.fft2(x)
        x_fft = self.conv(x_fft.real) + 1j * self.conv(x_fft.imag)
        
        if self.enforce:
            # Energy conservation constraint
            energy_in = torch.mean(torch.abs(x_fft)**2)
            x_fft = x_fft / torch.sqrt(energy_in + 1e-8)
            
        return torch.fft.ifft2(x_fft).real

class ModelSelector:
    def __init__(self, data_shape: tuple, data_type: str):
        self.data_shape = data_shape
        self.data_type = data_type
        self.available_models = {
            'fno': 'Fourier Neural Operator',
            'unet': 'U-Net with Fourier Blocks',
            'diffopt': 'Differentiable Optical Model',
            'convnet': 'Traditional CNN'
        }

    def get_user_preferences(self):
        """Collect user requirements through CLI prompts"""
        print("\n[Model Selection]")
        print("Available model types:")
        for key, value in self.available_models.items():
            print(f"{key}: {value}")
            
        self.model_choice = input("Select model architecture (fno/unet/diffopt/convnet): ").lower()
        self.priority = input("Optimize for (accuracy/speed/interpretability): ").lower()
        self.enforce_physics = input("Enforce physical constraints? (y/n): ").lower() == 'y'

    def build_model(self) -> nn.Module:
        """Construct the selected model architecture"""
        if self.model_choice == 'fno':
            return self._build_fno()
        elif self.model_choice == 'unet':
            return self._build_unet()
        elif self.model_choice == 'diffopt':
            return self._build_diffopt()
        else:
            return self._build_convnet()

    def _build_fno(self) -> nn.Module:
        """Fourier Neural Operator architecture"""
        return nn.Sequential(
            FourierBlock(1, 32, self.enforce_physics),
            nn.ReLU(),
            FourierBlock(32, 64, self.enforce_physics),
            nn.ReLU(),
            nn.Conv2d(64, 1 if self.data_type == 'complex_field' else 2, 3, padding=1)
        )

    def _build_unet(self) -> nn.Module:
        """U-Net with Fourier blocks"""
        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.down1 = nn.Sequential(
                    FourierBlock(1, 32, self.enforce_physics),
                    nn.MaxPool2d(2)
                )
                # Add more layers as needed
        return UNet()

    def _build_diffopt(self) -> nn.Module:
        """Differentiable optical model"""
        class DiffOptModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.propagate = nn.Sequential(
                    FourierBlock(1, 1, False),
                    nn.Parameter(torch.randn(1))  # Learnable propagation distance
                )
            def forward(self, x):
                return self.propagate(x)
        return DiffOptModel()

    def _build_convnet(self) -> nn.Module:
        """Traditional CNN baseline"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2 if self.data_type == 'complex_field' else 1, 3, padding=1)
        )

    def auto_recommend(self) -> str:
        """Automatically recommend model based on data characteristics"""
        if self.data_type == 'complex_field':
            return 'fno' if self.data_shape[0] > 256 else 'unet'
        elif self.data_type == 'psf':
            return 'diffopt'
        else:
            return 'convnet'

# Example usage
if __name__ == "__main__":
    selector = ModelSelector(data_shape=(256, 256), data_type='complex_field')
    selector.get_user_preferences()
    model = selector.build_model()
    print(f"\nBuilt model architecture:\n{model}")
