import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter
import os

class TrainingVisualizer:
    """Visualization tools for training progress"""
    
    def __init__(
        self,
        log_dir: str = 'logs',
        save_dir: str = 'visualizations'
    ):
        """
        Initialize visualizer
        
        Args:
            log_dir: Directory for TensorBoard logs
            save_dir: Directory for saving visualizations
        """
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int
    ):
        """
        Log metrics to TensorBoard
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f'Metrics/{name}', value, step)
    
    def log_phase_prediction(
        self,
        intensity: torch.Tensor,
        predicted_phase: torch.Tensor,
        target_phase: Optional[torch.Tensor] = None,
        step: int = 0
    ):
        """
        Log phase prediction visualization
        
        Args:
            intensity: Input intensity image
            predicted_phase: Predicted phase
            target_phase: Optional target phase for comparison
            step: Current step/epoch
        """
        # Convert tensors to numpy arrays
        intensity = intensity.detach().cpu().numpy()
        predicted_phase = predicted_phase.detach().cpu().numpy()
        if target_phase is not None:
            target_phase = target_phase.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3 if target_phase is not None else 2, figsize=(15, 5))
        
        # Plot intensity
        im0 = axes[0].imshow(intensity[0, 0], cmap='gray')
        axes[0].set_title('Intensity')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot predicted phase
        im1 = axes[1].imshow(predicted_phase[0, 0], cmap='viridis')
        axes[1].set_title('Predicted Phase')
        plt.colorbar(im1, ax=axes[1])
        
        # Plot target phase if available
        if target_phase is not None:
            im2 = axes[2].imshow(target_phase[0, 0], cmap='viridis')
            axes[2].set_title('Target Phase')
            plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.save_dir, f'phase_prediction_{step}.png')
        plt.savefig(fig_path)
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Phase_Prediction', fig, step)
    
    def log_optimization_history(
        self,
        history: List[Dict[str, float]],
        metric_name: str = 'loss'
    ):
        """
        Log optimization history
        
        Args:
            history: List of dictionaries containing metric values
            metric_name: Name of metric to plot
        """
        # Extract metric values
        values = [h[metric_name] for h in history]
        steps = range(len(values))
        
        # Create figure
        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, 'b-', label=metric_name)
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'Optimization History - {metric_name}')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.save_dir, f'optimization_history_{metric_name}.png')
        plt.savefig(fig_path)
        plt.close()
        
        # Log to TensorBoard
        for step, value in enumerate(values):
            self.writer.add_scalar(f'Optimization/{metric_name}', value, step)
    
    def log_model_architecture(
        self,
        model: torch.nn.Module,
        input_size: Tuple[int, int, int]
    ):
        """
        Log model architecture
        
        Args:
            model: PyTorch model
            input_size: Input size (channels, height, width)
        """
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Log model graph
        self.writer.add_graph(model, dummy_input)
    
    def log_parameter_distributions(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Log parameter distributions
        
        Args:
            model: PyTorch model
            step: Current step/epoch
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param.data, step)
    
    def log_gradient_distributions(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Log gradient distributions
        
        Args:
            model: PyTorch model
            step: Current step/epoch
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, step)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

def plot_phase_mask(
    phase_mask: np.ndarray,
    title: str = 'Phase Mask',
    save_path: Optional[str] = None
):
    """
    Plot phase mask
    
    Args:
        phase_mask: Phase mask array
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(phase_mask, cmap='viridis')
    plt.colorbar(label='Phase (rad)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_intensity(
    intensity: np.ndarray,
    title: str = 'Intensity',
    save_path: Optional[str] = None
):
    """
    Plot intensity
    
    Args:
        intensity: Intensity array
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(intensity, cmap='gray')
    plt.colorbar(label='Intensity (a.u.)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_comparison(
    original: np.ndarray,
    predicted: np.ndarray,
    title: str = 'Comparison',
    save_path: Optional[str] = None
):
    """
    Plot comparison between original and predicted
    
    Args:
        original: Original array
        predicted: Predicted array
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(predicted, cmap='gray')
    ax2.set_title('Predicted')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close() 