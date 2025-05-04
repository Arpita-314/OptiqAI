import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import optuna

def find_best_cnn(data, targets, n_trials=10):
    """Find the best CNN architecture using AutoML"""
    
    def create_model(trial):
        """Create a model with trial parameters"""
        n_layers = trial.suggest_int('n_layers', 2, 5)
        n_channels = [trial.suggest_int(f'n_channels_{i}', 16, 256) for i in range(n_layers)]
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        layers = []
        in_channels = 1  # Grayscale images
        
        for out_channels in n_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        model = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(n_channels[-1] * (data.shape[1] // (2**n_layers)) * (data.shape[2] // (2**n_layers)), 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, len(set(targets)))
        )
        
        return model
    
    def objective(trial):
        """Objective function to minimize"""
        model = create_model(trial)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-5, 1e-2))
        criterion = nn.CrossEntropyLoss()
        
        # Split data
        train_size = int(0.8 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        
        # Train for a few epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        for epoch in range(3):  # Quick training to evaluate architecture
            # Training
            model.train()
            for batch_data, batch_targets in train_loader:
                batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                    outputs = model(batch_data)
                    val_loss += criterion(outputs, batch_targets).item()
                    _, predicted = outputs.max(1)
                    total += batch_targets.size(0)
                    correct += predicted.eq(batch_targets).sum().item()
            
            accuracy = correct / total
        
        return val_loss / len(val_loader)  # Return validation loss
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Create and return the best model
    best_model = create_model(study.best_trial)
    return best_model