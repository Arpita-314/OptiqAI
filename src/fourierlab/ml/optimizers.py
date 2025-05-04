import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, List, Tuple
import math

class FourierOptimizer:
    """Advanced optimizer with Fourier-specific techniques"""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        nesterov: bool = True,
        amsgrad: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.amsgrad = amsgrad
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps,
            amsgrad=amsgrad
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=lr * 0.1
        )
    
    def step(self) -> None:
        """Perform optimization step"""
        self.optimizer.step()
        self.scheduler.step()
    
    def zero_grad(self) -> None:
        """Zero out gradients"""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.scheduler.get_last_lr()[0]

class FourierGradientClipping:
    """Gradient clipping with Fourier-specific thresholds"""
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        phase_clip: float = math.pi,
        magnitude_clip: float = 1.0
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.phase_clip = phase_clip
        self.magnitude_clip = magnitude_clip
    
    def clip_gradients(
        self,
        model: nn.Module
    ) -> None:
        """Clip gradients with Fourier-specific constraints"""
        # Clip overall gradient norm
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            self.norm_type
        )
        
        # Clip phase and magnitude gradients
        for param in model.parameters():
            if param.grad is not None:
                # Convert to Fourier domain
                grad_fft = torch.fft.fft2(param.grad)
                
                # Clip phase
                phase = torch.angle(grad_fft)
                phase = torch.clamp(phase, -self.phase_clip, self.phase_clip)
                
                # Clip magnitude
                magnitude = torch.abs(grad_fft)
                magnitude = torch.clamp(magnitude, 0, self.magnitude_clip)
                
                # Combine
                param.grad = torch.fft.ifft2(
                    magnitude * torch.exp(1j * phase)
                ).real

class FourierLearningRateFinder:
    """Learning rate finder with Fourier-specific analysis"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: FourierOptimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.lrs = []
        self.losses = []
        self.best_lr = None
    
    def find_lr(
        self,
        train_loader: torch.utils.data.DataLoader,
        init_lr: float = 1e-7,
        final_lr: float = 10,
        beta: float = 0.98
    ) -> Tuple[List[float], List[float]]:
        """Find optimal learning rate"""
        num = len(train_loader) - 1
        mult = (final_lr / init_lr) ** (1 / num)
        
        lr = init_lr
        for param_group in self.optimizer.optimizer.param_groups:
            param_group['lr'] = lr
        
        avg_loss = 0
        best_loss = 0
        batch_num = 0
        losses = []
        log_lrs = []
        
        for data in train_loader:
            batch_num += 1
            self.model.train()
            
            # Get data
            inputs, targets = data
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Compute smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            
            # Stop if loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            
            # Record best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            
            # Store values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            lr *= mult
            for param_group in self.optimizer.optimizer.param_groups:
                param_group['lr'] = lr
        
        return log_lrs, losses
    
    def plot_lr_finder(
        self,
        log_lrs: List[float],
        losses: List[float]
    ) -> None:
        """Plot learning rate finder results"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(log_lrs, losses)
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.show()

class FourierWeightDecay:
    """Weight decay with Fourier-specific regularization"""
    
    def __init__(
        self,
        model: nn.Module,
        weight_decay: float = 1e-4,
        phase_decay: float = 1e-3,
        magnitude_decay: float = 1e-4
    ):
        self.model = model
        self.weight_decay = weight_decay
        self.phase_decay = phase_decay
        self.magnitude_decay = magnitude_decay
    
    def compute_regularization(self) -> torch.Tensor:
        """Compute Fourier-specific regularization"""
        reg_loss = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                # Standard weight decay
                reg_loss += self.weight_decay * torch.norm(param)
                
                # Convert to Fourier domain
                param_fft = torch.fft.fft2(param)
                
                # Phase regularization
                phase = torch.angle(param_fft)
                reg_loss += self.phase_decay * torch.norm(phase)
                
                # Magnitude regularization
                magnitude = torch.abs(param_fft)
                reg_loss += self.magnitude_decay * torch.norm(magnitude)
        
        return reg_loss
