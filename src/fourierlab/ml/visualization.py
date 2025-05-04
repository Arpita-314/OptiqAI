import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from .pinn import PINN

class PINNVisualizer:
    """Visualization tools for Physics-Informed Neural Networks"""
    
    def __init__(
        self,
        model: PINN,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize PINN visualizer
        
        Args:
            model: PINN model
            device: Device to use for visualization
        """
        self.model = model.to(device)
        self.device = device
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """
        Plot training metrics
        
        Args:
            metrics: Dictionary of training metrics
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot total losses
        ax1.plot(metrics['train_loss'], label='Train Loss')
        ax1.plot(metrics['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Losses')
        ax1.legend()
        ax1.grid(True)
        
        # Plot physics losses
        for loss_name, loss_values in metrics['physics_losses'].items():
            if loss_values:  # Only plot if there are values
                ax2.plot(loss_values, label=loss_name)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Physics-Based Losses')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_field(
        self,
        field: torch.Tensor,
        title: str = 'Optical Field',
        save_path: Optional[str] = None
    ):
        """
        Plot optical field
        
        Args:
            field: Complex optical field
            title: Plot title
            save_path: Optional path to save plot
        """
        # Convert to numpy
        field_np = field.detach().cpu().numpy()
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot amplitude
        im1 = ax1.imshow(np.abs(field_np[0, 0]), cmap='viridis')
        ax1.set_title('Amplitude')
        plt.colorbar(im1, ax=ax1)
        
        # Plot phase
        im2 = ax2.imshow(np.angle(field_np[0, 0]), cmap='twilight')
        ax2.set_title('Phase')
        plt.colorbar(im2, ax=ax2)
        
        # Plot intensity
        im3 = ax3.imshow(np.abs(field_np[0, 0])**2, cmap='hot')
        ax3.set_title('Intensity')
        plt.colorbar(im3, ax=ax3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_propagation(
        self,
        field: torch.Tensor,
        distances: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot field propagation
        
        Args:
            field: Initial complex field
            distances: List of propagation distances
            save_path: Optional path to save plot
        """
        n_distances = len(distances)
        fig, axes = plt.subplots(2, n_distances, figsize=(5*n_distances, 10))
        
        # Propagate and plot
        for i, distance in enumerate(distances):
            # Propagate field
            propagated = self.model.propagator.angular_spectrum_propagate(
                field,
                distance=distance,
                wavelength=self.model.wavelength,
                pixel_size=self.model.pixel_size
            )
            
            # Convert to numpy
            field_np = propagated.detach().cpu().numpy()
            
            # Plot amplitude
            im1 = axes[0, i].imshow(np.abs(field_np[0, 0]), cmap='viridis')
            axes[0, i].set_title(f'Amplitude\nz = {distance:.3f}m')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Plot phase
            im2 = axes[1, i].imshow(np.angle(field_np[0, 0]), cmap='twilight')
            axes[1, i].set_title(f'Phase\nz = {distance:.3f}m')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.suptitle('Field Propagation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_physics_constraints(
        self,
        field: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Plot physics constraints
        
        Args:
            field: Complex optical field
            save_path: Optional path to save plot
        """
        # Calculate physics losses
        _, physics_losses = self.model(field)
        
        # Convert to numpy
        field_np = field.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot Poynting vector
        E = field
        H = torch.fft.fft2(E)
        S_x = torch.real(E * torch.conj(H))
        S_y = torch.imag(E * torch.conj(H))
        
        axes[0, 0].quiver(
            S_x[0, 0].detach().cpu().numpy(),
            S_y[0, 0].detach().cpu().numpy(),
            scale=50
        )
        axes[0, 0].set_title('Poynting Vector')
        
        # Plot angular momentum
        r = torch.meshgrid(
            torch.linspace(-1, 1, field.shape[-2]),
            torch.linspace(-1, 1, field.shape[-1])
        )
        r = torch.stack(r, dim=0).to(field.device)
        p = torch.abs(field)**2
        L = torch.cross(r, p)
        
        axes[0, 1].imshow(L[0, 0].detach().cpu().numpy(), cmap='viridis')
        axes[0, 1].set_title('Angular Momentum')
        
        # Plot polarization
        I = torch.abs(field)**2
        Q = torch.real(field * torch.conj(field))
        U = torch.imag(field * torch.conj(field))
        V = torch.angle(field)
        
        axes[0, 2].imshow(
            np.sqrt(Q[0, 0].detach().cpu().numpy()**2 + 
                   U[0, 0].detach().cpu().numpy()**2),
            cmap='viridis'
        )
        axes[0, 2].set_title('Polarization')
        
        # Plot wavefront curvature
        phase = torch.angle(field)
        dx = self.model.pixel_size
        dy = self.model.pixel_size
        
        d2x = (phase[:, :, :, 2:] - 2 * phase[:, :, :, 1:-1] + phase[:, :, :, :-2]) / (dx**2)
        d2y = (phase[:, :, 2:, :] - 2 * phase[:, :, 1:-1, :] + phase[:, :, :-2, :]) / (dy**2)
        curvature = d2x + d2y
        
        axes[1, 0].imshow(curvature[0, 0].detach().cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title('Wavefront Curvature')
        
        # Plot Helmholtz equation
        k = 2 * np.pi / self.model.wavelength
        helmholtz = d2x + d2y + (k**2) * field[:, :, 1:-1, 1:-1]
        
        axes[1, 1].imshow(
            torch.abs(helmholtz[0, 0]).detach().cpu().numpy(),
            cmap='viridis'
        )
        axes[1, 1].set_title('Helmholtz Equation')
        
        # Plot energy conservation
        energy = torch.abs(field)**2
        energy_grad = torch.abs(energy[1:] - energy[:-1])
        
        axes[1, 2].imshow(energy_grad[0, 0].detach().cpu().numpy(), cmap='viridis')
        axes[1, 2].set_title('Energy Conservation')
        
        plt.suptitle('Physics Constraints')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_material_interaction(
        self,
        field: torch.Tensor,
        material_properties: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot field interaction with materials
        
        Args:
            field: Complex optical field
            material_properties: Material properties
            save_path: Optional path to save plot
        """
        # Calculate Maxwell's equations losses
        maxwell_losses = self.model._maxwell_losses(field, material_properties)
        
        # Convert to numpy
        field_np = field.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot field in material
        axes[0, 0].imshow(np.abs(field_np[0, 0]), cmap='viridis')
        axes[0, 0].set_title('Field Amplitude in Material')
        
        # Plot material properties
        if 'permittivity' in material_properties:
            axes[0, 1].imshow(
                material_properties['permittivity'],
                cmap='viridis'
            )
            axes[0, 1].set_title('Material Permittivity')
        
        # Plot Maxwell's equations losses
        for i, (loss_name, loss_value) in enumerate(maxwell_losses.items()):
            axes[1, i].imshow(
                loss_value.detach().cpu().numpy(),
                cmap='viridis'
            )
            axes[1, i].set_title(f'{loss_name} Loss')
        
        plt.suptitle('Material Interaction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_training_animation(
        self,
        field: torch.Tensor,
        num_frames: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Create animation of training process
        
        Args:
            field: Initial complex field
            num_frames: Number of animation frames
            save_path: Optional path to save animation
        """
        import matplotlib.animation as animation
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize plots
        im1 = ax1.imshow(np.abs(field[0, 0].detach().cpu().numpy()), cmap='viridis')
        ax1.set_title('Field Amplitude')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(np.angle(field[0, 0].detach().cpu().numpy()), cmap='twilight')
        ax2.set_title('Field Phase')
        plt.colorbar(im2, ax=ax2)
        
        def update(frame):
            # Forward pass
            predicted_field, _ = self.model(field)
            
            # Update plots
            im1.set_array(np.abs(predicted_field[0, 0].detach().cpu().numpy()))
            im2.set_array(np.angle(predicted_field[0, 0].detach().cpu().numpy()))
            
            return [im1, im2]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=100,
            blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        plt.show() 