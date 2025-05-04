import torch
import optuna
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
from copy import deepcopy

# Add this import at the top of the file
from utils.metrics import FourierOpticsMetrics

class OpticsTrainer:
    def __init__(self, model: torch.nn.Module, wavelength: float, pixel_size: float, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.best_weights = None
        self.study = None
        self.metrics = FourierOpticsMetrics(wavelength, pixel_size)

    # ... (previous methods remain the same)

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {metric: 0.0 for metric in ["loss", "strehl_ratio", "mtf_correlation", "RMS_error", "PV_error", "phase_rmse"]}
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                
                total_metrics["loss"] += self.complex_loss(outputs, targets).item()
                batch_metrics = self.metrics.calculate_all_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    total_metrics[key] += value
        
        # Average the metrics
        return {key: value / len(loader) for key, value in total_metrics.items()}

    def manual_train(self, train_loader: DataLoader, val_loader: DataLoader, 
                     config: Dict[str, Any]):
        optimizer = torch.optim.AdamW(self.model.parameters())
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f}")
            for key, value in val_metrics.items():
                print(f"  Val {key}: {value:.4f}")
            
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                patience_counter = 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print("Early stopping triggered")
                    break
        
        self.model.load_state_dict(self.best_weights)

class OpticsTrainer:
    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.best_weights = None
        self.study = None

    def create_dataloader(self, data: torch.Tensor, targets: torch.Tensor, 
                         batch_size: int = 8, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from processed data"""
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def complex_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Custom loss for complex field reconstruction"""
        amp_loss = torch.nn.functional.mse_loss(torch.abs(pred), torch.abs(target))
        phase_loss = 1 - torch.cos(torch.angle(pred) - torch.angle(target)).mean()
        return amp_loss + 0.5 * phase_loss

    def energy_conservation_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Physics-informed regularization term"""
        energy_in = torch.sum(torch.abs(pred[0])**2)
        energy_out = torch.sum(torch.abs(pred[-1])**2)
        return torch.nn.functional.mse_loss(energy_out, energy_in)

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   lambda_physics: float = 0.1) -> float:
        self.model.train()
        total_loss = 0.0
        
        for data, targets in loader:
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
        """Interactive training configuration"""
        print("\n[Training Configuration]")
        return {
            "epochs": int(input("Max epochs (default 100): ") or 100),
            "patience": int(input("Early stopping patience (default 10): ") or 10),
            "batch_size": int(input("Batch size (default 8): ") or 8),
            "auto_tune": input("Run hyperparameter optimization? (y/n): ").lower() == "y"
        }

    def objective(self, trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader) -> float:
        """Optuna optimization objective"""
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
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader), 
            n_trials=n_trials
        )
        self.model.load_state_dict(self.best_weights)

    def manual_train(self, train_loader: DataLoader, val_loader: DataLoader, 
                    config: Dict[str, Any]):
        optimizer = torch.optim.AdamW(self.model.parameters())
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, phase_rmse = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Phase RMSE {phase_rmse:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.best_weights = deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print("Early stopping triggered")
                    break
        
        self.model.load_state_dict(self.best_weights)

# Example usage in main.py
def main():
    # ... previous code ...
    
    wavelength = 632.8e-9  # He-Ne laser wavelength (you might want to get this from user input)
    pixel_size = 5e-6  # 5 µm pixel size (you might want to get this from user input)
    
    trainer = OpticsTrainer(model, wavelength, pixel_size)
    
    # ... (rest of the main function)

    print("Next steps: Training and optimization")
    
    # Training setup
    from training.trainer import OpticsTrainer
    
    # Create sample data
    train_data = torch.randn(100, 1, 256, 256)  # Batch, channels, height, width
    train_targets = torch.randn(100, 2, 256, 256)  # Amplitude + phase
    val_data = torch.randn(20, 1, 256, 256)
    val_targets = torch.randn(20, 2, 256, 256)
    
    trainer = OpticsTrainer(model)
    train_loader = trainer.create_dataloader(train_data, train_targets, batch_size=8)
    val_loader = trainer.create_dataloader(val_data, val_targets, batch_size=8, shuffle=False)
    
    # Get user settings
    config = trainer.get_user_settings()
    
    if config["auto_tune"]:
        print("Running hyperparameter optimization...")
        trainer.auto_tune(train_loader, val_loader)
    else:
        print("Starting manual training...")
        trainer.manual_train(train_loader, val_loader, config)
    
    print("Training complete. Best model weights saved.")
    print("Next steps: Validation and deployment")

if __name__ == "__main__":
    main()
