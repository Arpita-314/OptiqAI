import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import optuna
from optuna.trial import Trial
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .models import FourierCNN, FourierUNet

class ArchitectureSearch:
    """Enhanced architecture search with advanced optimization techniques"""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_trials: int = 50,
        timeout: int = 3600,
        metric: str = "val_loss"
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        
        # Initialize study
        self.study = optuna.create_study(
            direction="minimize" if metric == "val_loss" else "maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
    
    def suggest_model_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest model architecture parameters"""
        return {
            "base_channels": trial.suggest_int("base_channels", 16, 64),
            "num_blocks": trial.suggest_int("num_blocks", 2, 5),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "use_attention": trial.suggest_categorical("use_attention", [True, False]),
            "use_residual": trial.suggest_categorical("use_residual", [True, False])
        }
    
    def suggest_optimizer_params(self, trial: Trial) -> Dict[str, Any]:
        """Suggest optimizer parameters"""
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "optimizer": trial.suggest_categorical(
                "optimizer",
                ["adam", "adamw", "sgd"]
            )
        }
    
    def create_optimizer(
        self,
        model: nn.Module,
        params: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Create optimizer with suggested parameters"""
        if params["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )
        elif params["optimizer"] == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"]
            )
        else:  # sgd
            return torch.optim.SGD(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                momentum=0.9
            )
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(self.device), batch_targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data, batch_targets = batch_data.to(self.device), batch_targets.to(self.device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_targets.size(0)
                correct += predicted.eq(batch_targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def objective(self, trial: Trial) -> float:
        """Objective function for optimization"""
        # Create model
        model_params = self.suggest_model_params(trial)
        model = FourierCNN(
            in_channels=self.input_shape[0],
            num_classes=self.num_classes,
            **model_params
        ).to(self.device)
        
        # Create optimizer
        optimizer_params = self.suggest_optimizer_params(trial)
        optimizer = self.create_optimizer(model, optimizer_params)
        
        # Create criterion
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(10):  # Train for 10 epochs
            # Train
            train_loss = self.train_epoch(model, self.train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(model, self.val_loader, criterion)
            
            # Report metrics
            trial.report(val_loss, epoch)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return val_loss if self.metric == "val_loss" else val_acc
    
    def optimize(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Run architecture search"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best model
        best_params = self.study.best_params
        model_params = {
            k: v for k, v in best_params.items()
            if k in ["base_channels", "num_blocks", "dropout_rate", "use_attention", "use_residual"]
        }
        
        best_model = FourierCNN(
            in_channels=self.input_shape[0],
            num_classes=self.num_classes,
            **model_params
        ).to(self.device)
        
        return best_model, best_params
    
    def get_best_model(self) -> nn.Module:
        """Get the best model from the study"""
        best_params = self.study.best_params
        model_params = {
            k: v for k, v in best_params.items()
            if k in ["base_channels", "num_blocks", "dropout_rate", "use_attention", "use_residual"]
        }
        
        return FourierCNN(
            in_channels=self.input_shape[0],
            num_classes=self.num_classes,
            **model_params
        ).to(self.device)
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history"""
        return {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_trials": len(self.study.trials),
            "best_trial": self.study.best_trial.number,
            "optimization_direction": self.study.direction
        }

def find_best_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    timeout: int = 3600,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Find the best CNN architecture for the given task.
    
    Args:
        input_shape: Shape of input data (channels, height, width)
        num_classes: Number of output classes
        train_loader: Training data loader
        val_loader: Validation data loader
        n_trials: Number of optimization trials
        timeout: Maximum time for optimization in seconds
        device: Device to use for training
        
    Returns:
        Tuple of (best model, best parameters)
    """
    searcher = ArchitectureSearch(
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        n_trials=n_trials,
        timeout=timeout
    )
    
    return searcher.optimize(train_loader, val_loader)