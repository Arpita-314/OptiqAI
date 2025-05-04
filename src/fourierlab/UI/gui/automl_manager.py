import optuna
import torch
import torch.nn as nn
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict, Any, Tuple

class AutoMLManager(QObject):
    # Signals
    trial_complete = pyqtSignal(dict)  # trial results
    optimization_complete = pyqtSignal(dict)  # best model and parameters
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.study = None
        self.best_model = None
        self.best_params = None
        
    def create_model(self, trial: optuna.Trial, input_size: Tuple[int, int]) -> nn.Module:
        """Create a model based on trial parameters"""
        # Define search space for model architecture
        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 5)
        n_filters = [trial.suggest_int(f'n_filters_{i}', 16, 128) for i in range(n_conv_layers)]
        kernel_sizes = [trial.suggest_int(f'kernel_size_{i}', 3, 7) for i in range(n_conv_layers)]
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        
        # Create model
        model = nn.Sequential()
        
        # Input layer
        in_channels = 1
        for i in range(n_conv_layers):
            model.add_module(f'conv_{i}', nn.Conv2d(
                in_channels, n_filters[i], kernel_sizes[i], padding=kernel_sizes[i]//2))
            if use_batch_norm:
                model.add_module(f'bn_{i}', nn.BatchNorm2d(n_filters[i]))
            model.add_module(f'relu_{i}', nn.ReLU())
            model.add_module(f'dropout_{i}', nn.Dropout2d(dropout_rate))
            in_channels = n_filters[i]
        
        # Output layer
        model.add_module('conv_out', nn.Conv2d(in_channels, 1, 1))
        model.add_module('tanh', nn.Tanh())
        
        return model
    
    def objective(self, trial: optuna.Trial, train_loader, val_loader, device) -> float:
        """Objective function for optimization"""
        # Create model
        model = self.create_model(trial, (256, 256))  # Assuming 256x256 input
        model = model.to(device)
        
        # Optimizer parameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training parameters
        n_epochs = trial.suggest_int('n_epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += nn.MSELoss()(output, target).item()
            
            # Report intermediate value
            trial.report(val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return best_val_loss
    
    def optimize(self, train_loader, val_loader, n_trials=50):
        """Run AutoML optimization"""
        try:
            # Create study
            self.study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Run optimization
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.study.optimize(
                lambda trial: self.objective(trial, train_loader, val_loader, device),
                n_trials=n_trials
            )
            
            # Get best model and parameters
            self.best_params = self.study.best_params
            self.best_model = self.create_model(
                optuna.trial.FixedTrial(self.best_params),
                (256, 256)  # Assuming 256x256 input
            )
            
            # Emit results
            results = {
                'best_params': self.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'model_summary': str(self.best_model)
            }
            self.optimization_complete.emit(results)
            
            return results
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            return None
    
    def get_best_model(self):
        """Get the best model found during optimization"""
        return self.best_model
    
    def get_best_params(self):
        """Get the best parameters found during optimization"""
        return self.best_params
    
    def get_trial_history(self):
        """Get the history of all trials"""
        if self.study is None:
            return []
        
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
        return history 