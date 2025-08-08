import torch
import optuna
#import pyyaml
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
from copy import deepcopy
import logging
import argparse
from tqdm import tqdm

# Add this import at the top of the file
from utils.metrics import FourierOpticsMetrics
class OpticsTrainer:
    """
    Trainer for physics-informed neural networks in Fourier optics.

    This class handles model training, validation, early stopping, 
    hyperparameter optimization (with Optuna), and model persistence.
    """

    def __init__(self, model: torch.nn.Module, wavelength: float = None, pixel_size: float = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the OpticsTrainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            wavelength (float, optional): Wavelength parameter for physics-informed loss.
            pixel_size (float, optional): Pixel size parameter for physics-informed loss.
            device (str, optional): Device to use ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.best_weights = None
        self.study = None

    def create_dataloader(self, data: torch.Tensor, targets: torch.Tensor, batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader from tensors.

        Args:
            data (torch.Tensor): Input data tensor.
            targets (torch.Tensor): Target data tensor.
            batch_size (int): Batch size for loading data.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader for the dataset.
        """
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def complex_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute a custom loss for complex field reconstruction.

        Args:
            pred (torch.Tensor): Predicted complex field.
            target (torch.Tensor): Ground truth complex field.

        Returns:
            torch.Tensor: Computed loss value.
        """
        amp_loss = torch.nn.functional.mse_loss(torch.abs(pred), torch.abs(target))
        phase_loss = 1 - torch.cos(torch.angle(pred) - torch.angle(target)).mean()
        return amp_loss + 0.5 * phase_loss

    def energy_conservation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Physics-informed regularization term enforcing energy conservation.

        Args:
            pred (torch.Tensor): Predicted complex field.

        Returns:
            torch.Tensor: Energy conservation loss.
        """
        energy_in = torch.sum(torch.abs(pred[0])**2)
        energy_out = torch.sum(torch.abs(pred[-1])**2)
        return torch.nn.functional.mse_loss(energy_out, energy_in)

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, lambda_physics: float = 0.1) -> float:
        """
        Train the model for one epoch.

        Args:
            loader (DataLoader): Training data loader.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            lambda_physics (float): Weight for physics-informed loss.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for data, targets in tqdm(loader, desc="Training"):
            data, targets = data.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            
            outputs = self.model(data)
            loss = self.complex_loss(outputs, targets)
            
            if lambda_physics > 0:
                physics_loss = self.energy_conservation_loss(outputs)
                loss += lambda_physics * physics_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            loader (DataLoader): Validation data loader.

        Returns:
            Tuple[float, float]: Validation loss and phase RMSE.
        """
        self.model.eval()
        total_loss = 0.0
        phase_rmse = 0.0
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                
                total_loss += self.complex_loss(outputs, targets).item()
                phase_rmse += torch.sqrt(torch.mean(
                    (torch.angle(outputs) - torch.angle(targets))**2
                )).item()
        
        return total_loss / len(loader), phase_rmse / len(loader)

    def get_user_settings(self) -> Dict[str, Any]:
        """
        Get training configuration from user input.

        Returns:
            Dict[str, Any]: Dictionary of training settings.
        """
        print("\n[Training Configuration]")
        return {
            "epochs": int(input("Max epochs (default 100): ") or 100),
            "patience": int(input("Early stopping patience (default 10): ") or 10),
            "batch_size": int(input("Batch size (default 8): ") or 8),
            "auto_tune": input("Run hyperparameter optimization? (y/n): ").lower() == "y"
        }

    def objective(self, trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader) -> float:
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.

        Returns:
            float: Best validation loss achieved.
        """
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        lambda_physics = trial.suggest_float("lambda_physics", 0.0, 1.0)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(100):
            train_loss = self.train_epoch(train_loader, optimizer, lambda_physics)
            val_loss, _ = self.validate(val_loader)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
        
        return best_val_loss

    def auto_tune(self, train_loader: DataLoader, val_loader: DataLoader, n_trials: int = 20):
        """
        Run Optuna hyperparameter optimization.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            n_trials (int): Number of Optuna trials.
        """
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader), 
            n_trials=n_trials
        )
        self.model.load_state_dict(self.best_weights)

    def manual_train(self, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]):
        """
        Train the model manually with early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            config (Dict[str, Any]): Training configuration.
        """
        optimizer = torch.optim.AdamW(self.model.parameters())
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, phase_rmse = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Phase RMSE {phase_rmse:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    logger.info("Early stopping triggered")
                    print("Early stopping triggered")
                    break
        
        self.model.load_state_dict(self.best_weights)

    def save_model(self, path: str):
        """
        Save the model weights to a file.

        Args:
            path (str): File path to save the model.
        """
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model weights saved to {path}")
        except Exception as e:
            logger.exception(f"Failed to save model to {path}: {e}")

    def load_model(self, path: str):
        """
        Load model weights from a file.

        Args:
            path (str): File path to load the model from.
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model weights loaded from {path}")
        except Exception as e:
            logger.exception(f"Failed to load model from {path}: {e}")

def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file {path} not found.")
        raise
    except Exception as e:
        logger.exception(f"Error loading config file {path}: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Example usage in main.py
def main():
    try:
        # Initialize model (replace with your actual model)
        model = MyModel()

        wavelength = 632.8e-9
        pixel_size = 5e-6

        trainer = OpticsTrainer(model, wavelength, pixel_size)

        logger.info("Next steps: Training and optimization")

        # Create sample data
        train_data = torch.randn(100, 1, 256, 256)
        train_targets = torch.randn(100, 2, 256, 256)
        val_data = torch.randn(20, 1, 256, 256)
        val_targets = torch.randn(20, 2, 256, 256)

        train_loader = trainer.create_dataloader(train_data, train_targets, batch_size=config["batch_size"])
        val_loader = trainer.create_dataloader(val_data, val_targets, batch_size=config["batch_size"], shuffle=False)

        config = trainer.get_user_settings()

        if config["auto_tune"]:
            logger.info("Running hyperparameter optimization...")
            print("Running hyperparameter optimization...")
            trainer.auto_tune(train_loader, val_loader)
        else:
            logger.info("Starting manual training...")
            print("Starting manual training...")
            trainer.manual_train(train_loader, val_loader, config)

        logger.info("Training complete. Best model weights saved.")
        print("Training complete. Best model weights saved.")
        logger.info("Next steps: Validation and deployment")
        print("Next steps: Validation and deployment")
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        raise

config = load_config("config.yaml")
# Use config["epochs"], config["batch_size"], etc.

if __name__ == "__main__":
    main()

import torch
from trainer import OpticsTrainer

def test_trainer_init():
    model = torch.nn.Linear(10, 2)
    trainer = OpticsTrainer(model)
    assert trainer.model is not None

epochs: 100
patience: 10
batch_size: 8
auto_tune: false
