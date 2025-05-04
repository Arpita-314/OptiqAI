import numpy as np
import torch
from scipy import signal
from typing import Tuple, Dict

class FourierOpticsMetrics:
    def __init__(self, wavelength: float, pixel_size: float):
        """
        Initialize the metrics calculator.
        
        Args:
        wavelength (float): Wavelength of light in meters
        pixel_size (float): Size of each pixel in meters
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size

    def strehl_ratio(self, wavefront: torch.Tensor) -> float:
        """
        Calculate the Strehl ratio.
        
        Args:
        wavefront (torch.Tensor): Complex wavefront

        Returns:
        float: Strehl ratio
        """
        intensity = torch.abs(wavefront)**2
        max_intensity = torch.max(intensity)
        ideal_intensity = torch.sum(intensity)
        return (max_intensity / ideal_intensity).item()

    def mtf(self, psf: torch.Tensor) -> np.ndarray:
        """
        Calculate the Modulation Transfer Function.
        
        Args:
        psf (torch.Tensor): Point Spread Function

        Returns:
        np.ndarray: 2D MTF
        """
        otf = torch.fft.fft2(psf)
        mtf = torch.abs(torch.fft.fftshift(otf))
        return mtf.cpu().numpy() / np.max(mtf.cpu().numpy())

    def wavefront_error_stats(self, wavefront: torch.Tensor) -> Dict[str, float]:
        """
        Calculate wavefront error statistics.
        
        Args:
        wavefront (torch.Tensor): Complex wavefront

        Returns:
        Dict[str, float]: Dictionary containing RMS and PV error
        """
        phase = torch.angle(wavefront)
        rms_error = torch.std(phase).item()
        pv_error = (torch.max(phase) - torch.min(phase)).item()
        return {"RMS_error": rms_error, "PV_error": pv_error}

    def calculate_all_metrics(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Args:
        predicted (torch.Tensor): Predicted complex field
        target (torch.Tensor): Target complex field

        Returns:
        Dict[str, float]: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Strehl Ratio
        metrics["strehl_ratio"] = self.strehl_ratio(predicted)
        
        # MTF correlation
        pred_mtf = self.mtf(torch.abs(predicted)**2)
        target_mtf = self.mtf(torch.abs(target)**2)
        metrics["mtf_correlation"] = np.corrcoef(pred_mtf.flatten(), target_mtf.flatten())[0, 1]
        
        # Wavefront error stats
        error_stats = self.wavefront_error_stats(predicted / target)
        metrics.update(error_stats)
        
        # Phase RMSE
        phase_rmse = torch.sqrt(torch.mean((torch.angle(predicted) - torch.angle(target))**2)).item()
        metrics["phase_rmse"] = phase_rmse
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Simulating some sample data
    wavelength = 632.8e-9  # He-Ne laser wavelength
    pixel_size = 5e-6  # 5 µm pixel size
    
    # Create sample wavefronts (you'd replace these with actual model outputs)
    target = torch.exp(1j * torch.randn(256, 256))
    predicted = torch.exp(1j * (torch.randn(256, 256) * 0.9 + 0.1))  # Slightly perturbed
    
    metrics_calculator = FourierOpticsMetrics(wavelength, pixel_size)
    results = metrics_calculator.calculate_all_metrics(predicted, target)
    
    print("Fourier Optics Metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
