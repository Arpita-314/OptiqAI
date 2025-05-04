import torch
import torch.nn as nn
import torch.cuda.amp as amp
from typing import Optional, Dict, Any, List, Tuple
import math

class MixedPrecisionTrainer:
    """Mixed precision training for Fourier models"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scaler: Optional[amp.GradScaler] = None,
        enabled: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = scaler or amp.GradScaler(enabled=enabled)
        self.enabled = enabled
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single training step with mixed precision"""
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with amp.autocast(enabled=self.enabled):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Get metrics
        metrics = {
            'loss': loss.item(),
            'scale': self.scaler.get_scale()
        }
        
        return loss, metrics
    
    def validate_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single validation step with mixed precision"""
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass with mixed precision
        with torch.no_grad(), amp.autocast(enabled=self.enabled):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Get metrics
        metrics = {
            'loss': loss.item()
        }
        
        return loss, metrics
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch with mixed precision"""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'scale': 0.0
        }
        
        for inputs, targets in train_loader:
            loss, metrics = self.train_step(inputs, targets)
            
            # Update metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
        
        # Average metrics
        num_batches = len(train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch with mixed precision"""
        self.model.eval()
        epoch_metrics = {
            'loss': 0.0
        }
        
        for inputs, targets in val_loader:
            loss, metrics = self.validate_step(inputs, targets)
            
            # Update metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
        
        # Average metrics
        num_batches = len(val_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_metrics

class MixedPrecisionGAN:
    """Mixed precision training for Fourier GAN"""
    
    def __init__(
        self,
        gan: nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scaler_g: Optional[amp.GradScaler] = None,
        scaler_d: Optional[amp.GradScaler] = None,
        enabled: bool = True
    ):
        self.gan = gan
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.criterion = criterion
        self.device = device
        self.scaler_g = scaler_g or amp.GradScaler(enabled=enabled)
        self.scaler_d = scaler_d or amp.GradScaler(enabled=enabled)
        self.enabled = enabled
    
    def train_step(
        self,
        real_inputs: torch.Tensor,
        latent_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Perform a single GAN training step with mixed precision"""
        # Move data to device
        real_inputs = real_inputs.to(self.device)
        
        # Generate fake inputs
        z = torch.randn(real_inputs.size(0), latent_dim, device=self.device)
        
        # Train discriminator
        self.optimizer_d.zero_grad()
        
        with amp.autocast(enabled=self.enabled):
            # Real inputs
            real_outputs = self.gan.discriminator(real_inputs)
            real_loss = self.criterion.discriminator_loss(
                real_outputs,
                torch.ones_like(real_outputs)
            )
            
            # Fake inputs
            fake_inputs = self.gan.generator(z)
            fake_outputs = self.gan.discriminator(fake_inputs.detach())
            fake_loss = self.criterion.discriminator_loss(
                fake_outputs,
                torch.zeros_like(fake_outputs)
            )
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss
        
        # Backward pass for discriminator
        self.scaler_d.scale(d_loss).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()
        
        # Train generator
        self.optimizer_g.zero_grad()
        
        with amp.autocast(enabled=self.enabled):
            # Generate new fake inputs
            fake_inputs = self.gan.generator(z)
            fake_outputs = self.gan.discriminator(fake_inputs)
            
            # Generator loss
            g_loss = self.criterion.generator_loss(
                fake_outputs,
                fake_inputs,
                real_inputs
            )
        
        # Backward pass for generator
        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        
        # Get metrics
        metrics = {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_scale': self.scaler_g.get_scale(),
            'd_scale': self.scaler_d.get_scale()
        }
        
        return g_loss, d_loss, metrics
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        latent_dim: int
    ) -> Dict[str, float]:
        """Train GAN for one epoch with mixed precision"""
        self.gan.train()
        epoch_metrics = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'g_scale': 0.0,
            'd_scale': 0.0
        }
        
        for real_inputs, _ in train_loader:
            g_loss, d_loss, metrics = self.train_step(real_inputs, latent_dim)
            
            # Update metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
        
        # Average metrics
        num_batches = len(train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_metrics

class MixedPrecisionScheduler:
    """Learning rate scheduler with mixed precision support"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: amp.GradScaler
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
    
    def step(self) -> None:
        """Step the scheduler with mixed precision support"""
        self.scheduler.step()
        
        # Update scaler if needed
        if self.scaler.get_scale() < 1e-5:
            self.scaler.update(2.0)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]
    
    def get_scale(self) -> float:
        """Get current gradient scale"""
        return self.scaler.get_scale() 