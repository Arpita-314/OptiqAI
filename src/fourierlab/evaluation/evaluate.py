import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torch.nn.functional as F

from ..models.phase_retrieval import PhaseCNN
from ..utils.dataset import PhaseRetrievalDataset
from ..utils.visualization import TrainingVisualizer, plot_comparison
from ..physics.propagator import WavePropagator

def evaluate_model(
    model: PhaseCNN,
    test_loader: DataLoader,
    device: str,
    save_dir: str = 'evaluation_results'
) -> Dict[str, float]:
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'phase_mse': 0.0,
        'phase_mae': 0.0,
        'intensity_mse': 0.0,
        'intensity_mae': 0.0,
        'psnr': 0.0,
        'ssim': 0.0
    }
    
    # Initialize wave propagator
    propagator = WavePropagator()
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(
        log_dir=os.path.join(save_dir, 'logs'),
        save_dir=os.path.join(save_dir, 'visualizations')
    )
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            # Get data
            intensity = batch['intensity'].to(device)
            target_phase = batch['phase'].to(device)
            wavelength = batch['wavelength'].to(device)
            pixel_size = batch['pixel_size'].to(device)
            
            # Get prediction
            predicted_phase = model.predict_phase(
                intensity,
                wavelength=wavelength,
                pixel_size=pixel_size
            )
            
            # Calculate intensity from predicted phase
            field = torch.exp(1j * predicted_phase)
            predicted_intensity = propagator.calculate_intensity(field)
            predicted_intensity = predicted_intensity / predicted_intensity.max()
            
            # Calculate metrics
            phase_mse = torch.mean((predicted_phase - target_phase)**2)
            phase_mae = torch.mean(torch.abs(predicted_phase - target_phase))
            intensity_mse = torch.mean((predicted_intensity - intensity)**2)
            intensity_mae = torch.mean(torch.abs(predicted_intensity - intensity))
            
            # Calculate PSNR
            mse = torch.mean((predicted_intensity - intensity)**2)
            psnr = 10 * torch.log10(1.0 / mse)
            
            # Calculate SSIM
            ssim = calculate_ssim(predicted_intensity, intensity)
            
            # Update metrics
            metrics['phase_mse'] += phase_mse.item()
            metrics['phase_mae'] += phase_mae.item()
            metrics['intensity_mse'] += intensity_mse.item()
            metrics['intensity_mae'] += intensity_mae.item()
            metrics['psnr'] += psnr.item()
            metrics['ssim'] += ssim.item()
            
            # Save visualization
            if i < 10:  # Save first 10 examples
                plot_comparison(
                    intensity[0, 0].cpu().numpy(),
                    predicted_intensity[0, 0].cpu().numpy(),
                    title=f'Example {i+1}',
                    save_path=os.path.join(save_dir, 'predictions', f'example_{i+1}.png')
                )
    
    # Average metrics
    n_samples = len(test_loader)
    for key in metrics:
        metrics[key] /= n_samples
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value:.6f}\n')
    
    return metrics

def calculate_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        x: First image
        y: Second image
        window_size: Size of Gaussian window
        
    Returns:
        SSIM value
    """
    # Constants
    C1 = 0.01**2
    C2 = 0.03**2
    
    # Create Gaussian window
    window = torch.exp(-torch.arange(-(window_size//2), window_size//2+1)**2 / (2 * (window_size/6)**2))
    window = window / window.sum()
    window = window.view(1, 1, -1, 1)  # Shape: [1, 1, window_size, 1]
    window = window * window.transpose(-1, -2)  # Shape: [1, 1, window_size, window_size]
    window = window.to(x.device)
    
    # Compute means
    mu_x = F.conv2d(x, window, padding=window_size//2)
    mu_y = F.conv2d(y, window, padding=window_size//2)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    
    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x**2, window, padding=window_size//2) - mu_x_sq
    sigma_y_sq = F.conv2d(y**2, window, padding=window_size//2) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size//2) - mu_xy
    
    # Compute SSIM
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim = numerator / denominator
    
    return ssim.mean()

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate phase retrieval model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to evaluate on')
    args = parser.parse_args()
    
    # Load model
    model = PhaseCNN.load(args.model_path)
    model = model.to(args.device)
    
    # Create test dataset and loader
    test_dataset = PhaseRetrievalDataset(
        data_dir=args.data_dir,
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # Print metrics
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

if __name__ == '__main__':
    main() 