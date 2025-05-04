import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

class FourierOpticsVisualization:
    def __init__(self):
        pass

    def plot_intensity(self, intensity: torch.Tensor, title: str = "Intensity", save_path: str = None):
        """
        Plot the intensity distribution.

        Args:
        intensity (torch.Tensor): 2D intensity data
        title (str): Title of the plot
        save_path (str): Path to save the plot (optional)
        """
        intensity = intensity.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(intensity, cmap="hot", extent=[0, intensity.shape[1], 0, intensity.shape[0]])
        plt.colorbar(label="Intensity")
        plt.title(title)
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        plt.show()

    def plot_phase(self, phase: torch.Tensor, title: str = "Phase", save_path: str = None):
        """
        Plot the phase map.

        Args:
        phase (torch.Tensor): 2D phase data
        title (str): Title of the plot
        save_path (str): Path to save the plot (optional)
        """
        phase = phase.cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(phase, cmap="twilight", extent=[0, phase.shape[1], 0, phase.shape[0]])
        plt.colorbar(label="Phase (radians)")
        plt.title(title)
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        plt.show()

    def plot_mtf(self, mtf: np.ndarray, title: str = "Modulation Transfer Function", save_path: str = None):
        """
        Plot the Modulation Transfer Function.

        Args:
        mtf (np.ndarray): 2D MTF data
        title (str): Title of the plot
        save_path (str): Path to save the plot (optional)
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(mtf, cmap="viridis", extent=[-1, 1, -1, 1])
        plt.colorbar(label="MTF")
        plt.title(title)
        plt.xlabel("Spatial Frequency X")
        plt.ylabel("Spatial Frequency Y")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        plt.show()

    def compare_wavefronts(self, predicted: torch.Tensor, target: torch.Tensor, save_dir: str = None):
        """
        Compare predicted and target wavefronts by plotting amplitude and phase.

        Args:
        predicted (torch.Tensor): Predicted complex wavefront
        target (torch.Tensor): Target complex wavefront
        save_dir (str): Directory to save plots (optional)
        """
        
        # Amplitude comparison
        pred_amplitude = torch.abs(predicted)
        target_amplitude = torch.abs(target)
        
        self.plot_intensity(pred_amplitude, title="Predicted Amplitude", 
                            save_path=f"{save_dir}/predicted_amplitude.png" if save_dir else None)
        
        self.plot_intensity(target_amplitude, title="Target Amplitude", 
                            save_path=f"{save_dir}/target_amplitude.png" if save_dir else None)
        
        # Phase comparison
        pred_phase = torch.angle(predicted)
        target_phase = torch.angle(target)
        
        self.plot_phase(pred_phase, title="Predicted Phase", 
                        save_path=f"{save_dir}/predicted_phase.png" if save_dir else None)
        
        self.plot_phase(target_phase, title="Target Phase", 
                        save_path=f"{save_dir}/target_phase.png" if save_dir else None)

# Example usage
if __name__ == "__main__":
    # Simulated wavefronts for testing visualization
    predicted_wavefront = torch.exp(1j * torch.randn(256, 256))
    target_wavefront = torch.exp(1j * (torch.randn(256, 256) * 0.9 + 0.1))  # Slightly perturbed
    
    viz = FourierOpticsVisualization()
    
    # Compare wavefronts
    viz.compare_wavefronts(predicted_wavefront, target_wavefront)

    # Plot MTF example
    mtf_example = np.random.rand(256, 256)  # Replace with actual MTF data
    viz.plot_mtf(mtf_example)
