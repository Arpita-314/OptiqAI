import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from ..models.phase_retrieval import PhaseCNN
from ..physics.propagator import WavePropagator
from ..utils.dataset import PhaseRetrievalDataset

class PhaseRetrievalTrainer:
    """Trainer for phase retrieval model"""
    
    def __init__(
        self,
        model: PhaseCNN,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Phase retrieval model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            device: Device to train on
            config: Optional training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Default configuration
        self.config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'n_epochs': 100,
            'save_dir': 'checkpoints',
            'log_dir': 'logs',
            'save_freq': 10,
            'val_freq': 1,
            'early_stopping_patience': 10
        }
        if config:
            self.config.update(config)
        
        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # Initialize wave propagator for physical constraints
        self.propagator = WavePropagator()
        
        # Initialize loss functions
        self.phase_loss = nn.MSELoss()
        self.intensity_loss = nn.MSELoss()
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Get data
            intensity = batch['intensity'].to(self.device)
            target_phase = batch['phase'].to(self.device)
            wavelength = batch['wavelength'].to(self.device)
            pixel_size = batch['pixel_size'].to(self.device)
            
            # Forward pass
            predicted_phase = self.model.predict_phase(
                intensity,
                wavelength=wavelength,
                pixel_size=pixel_size
            )
            
            # Calculate losses
            phase_loss = self.phase_loss(predicted_phase, target_phase)
            
            # Calculate intensity from predicted phase
            predicted_intensity = self.propagator.calculate_intensity(
                torch.exp(1j * predicted_phase)
            )
            intensity_loss = self.intensity_loss(predicted_intensity, intensity)
            
            # Total loss
            loss = phase_loss + intensity_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        if not self.val_loader:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Get data
                intensity = batch['intensity'].to(self.device)
                target_phase = batch['phase'].to(self.device)
                wavelength = batch['wavelength'].to(self.device)
                pixel_size = batch['pixel_size'].to(self.device)
                
                # Forward pass
                predicted_phase = self.model.predict_phase(
                    intensity,
                    wavelength=wavelength,
                    pixel_size=pixel_size
                )
                
                # Calculate losses
                phase_loss = self.phase_loss(predicted_phase, target_phase)
                
                # Calculate intensity from predicted phase
                predicted_intensity = self.propagator.calculate_intensity(
                    torch.exp(1j * predicted_phase)
                )
                intensity_loss = self.intensity_loss(predicted_intensity, intensity)
                
                # Total loss
                loss = phase_loss + intensity_loss
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train model"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['n_epochs']):
            # Train
            train_loss = self.train_epoch()
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Validate
            if epoch % self.config['val_freq'] == 0:
                val_loss = self.validate()
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.model.save(
                        os.path.join(self.config['save_dir'], 'best_model.pt')
                    )
                else:
                    patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config['save_freq'] == 0:
                self.model.save(
                    os.path.join(self.config['save_dir'], f'checkpoint_{epoch}.pt')
                )
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f'Early stopping at epoch {epoch}')
                break
        
        self.writer.close()

def main():
    """Main training function"""
    # Create dataset
    train_dataset = PhaseRetrievalDataset(
        data_dir='data/train',
        transform=None
    )
    val_dataset = PhaseRetrievalDataset(
        data_dir='data/val',
        transform=None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = PhaseCNN(
        input_size=(256, 256),
        n_channels=1,
        n_filters=32,
        n_layers=3
    )
    
    # Create trainer
    trainer = PhaseRetrievalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'n_epochs': 100,
            'save_dir': 'checkpoints',
            'log_dir': 'logs',
            'save_freq': 10,
            'val_freq': 1,
            'early_stopping_patience': 10
        }
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main() 