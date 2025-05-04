import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..physics.propagator import WavePropagator
from ..physics.maxwell import MaxwellEquations
from ..physics.materials import MaterialProperties
from ..physics.quantum import QuantumOptics

class PINN(nn.Module):
    """Physics-Informed Neural Network for Fourier optics"""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),
        hidden_dims: List[int] = [64, 128, 256, 128, 64],
        activation: nn.Module = nn.Tanh(),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        wavelength: float = 632.8e-9,
        pixel_size: float = 5e-6,
        material_properties: Optional[Dict] = None,
        use_maxwell: bool = True,
        use_advanced_physics: bool = True,
        use_quantum: bool = False,
        use_curriculum: bool = True
    ):
        """
        Initialize PINN for Fourier optics
        
        Args:
            input_size: Input image size (height, width)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            wavelength: Wavelength of light in meters
            pixel_size: Size of each pixel in meters
            material_properties: Optional material properties for Maxwell's equations
            use_maxwell: Whether to use Maxwell's equations
            use_advanced_physics: Whether to use advanced physics constraints
            use_quantum: Whether to use quantum optics
            use_curriculum: Whether to use curriculum learning
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.use_maxwell = use_maxwell
        self.use_advanced_physics = use_advanced_physics
        self.use_quantum = use_quantum
        self.use_curriculum = use_curriculum
        
        # Initialize physics models
        self.propagator = WavePropagator(
            wavelength=wavelength,
            pixel_size=pixel_size
        )
        
        if use_maxwell:
            self.maxwell = MaxwellEquations(
                wavelength=wavelength,
                pixel_size=pixel_size,
                material_properties=material_properties
            )
        
        if use_quantum:
            self.quantum = QuantumOptics(
                wavelength=wavelength,
                pixel_size=pixel_size
            )
        
        # Build network layers
        self.layers = nn.ModuleList()
        in_channels = 1  # Input is complex field (real + imag)
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                activation,
                nn.Dropout2d(dropout_rate)
            ))
            in_channels = hidden_dim
        
        # Final layer to predict complex field
        self.final_layer = nn.Conv2d(in_channels, 2, 1)  # 2 channels for real and imaginary parts
        
        # Curriculum learning parameters
        if use_curriculum:
            self.curriculum_stages = [
                {'epochs': 10, 'loss_weights': {
                    'helmholtz': 1.0,
                    'energy': 0.0,
                    'phase_continuity': 0.0,
                    'propagation': 0.0,
                    'maxwell': 0.0,
                    'quantum': 0.0
                }},
                {'epochs': 20, 'loss_weights': {
                    'helmholtz': 1.0,
                    'energy': 0.1,
                    'phase_continuity': 0.1,
                    'propagation': 0.0,
                    'maxwell': 0.0,
                    'quantum': 0.0
                }},
                {'epochs': 30, 'loss_weights': {
                    'helmholtz': 1.0,
                    'energy': 0.1,
                    'phase_continuity': 0.1,
                    'propagation': 1.0,
                    'maxwell': 0.0,
                    'quantum': 0.0
                }},
                {'epochs': 40, 'loss_weights': {
                    'helmholtz': 1.0,
                    'energy': 0.1,
                    'phase_continuity': 0.1,
                    'propagation': 1.0,
                    'maxwell': 1.0,
                    'quantum': 0.0
                }},
                {'epochs': 50, 'loss_weights': {
                    'helmholtz': 1.0,
                    'energy': 0.1,
                    'phase_continuity': 0.1,
                    'propagation': 1.0,
                    'maxwell': 1.0,
                    'quantum': 1.0
                }}
            ]
            self.current_stage = 0
    
    def forward(
        self,
        x: torch.Tensor,
        distance: Optional[float] = None,
        material_properties: Optional[Dict] = None,
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with physics constraints
        
        Args:
            x: Input complex field
            distance: Optional propagation distance
            material_properties: Optional material properties for Maxwell's equations
            epoch: Current training epoch for curriculum learning
            
        Returns:
            Tuple of (predicted field, physics losses)
        """
        # Process through network
        for layer in self.layers:
            x = layer(x)
        
        # Get real and imaginary parts
        real, imag = torch.chunk(self.final_layer(x), 2, dim=1)
        predicted_field = real + 1j * imag
        
        # Calculate physics losses
        physics_losses = self._calculate_physics_losses(
            predicted_field,
            distance,
            material_properties,
            epoch
        )
        
        return predicted_field, physics_losses
    
    def _calculate_physics_losses(
        self,
        field: torch.Tensor,
        distance: Optional[float],
        material_properties: Optional[Dict],
        epoch: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        """Calculate physics-based losses"""
        losses = {}
        
        # Get current curriculum stage
        if self.use_curriculum and epoch is not None:
            current_stage = self._get_curriculum_stage(epoch)
            loss_weights = current_stage['loss_weights']
        else:
            loss_weights = {
                'helmholtz': 1.0,
                'energy': 0.1,
                'phase_continuity': 0.1,
                'propagation': 1.0,
                'maxwell': 1.0,
                'quantum': 1.0
            }
        
        # Basic physics losses
        if loss_weights['helmholtz'] > 0:
            losses['helmholtz'] = self._helmholtz_loss(field)
        if loss_weights['energy'] > 0:
            losses['energy'] = self._energy_conservation_loss(field)
        if loss_weights['phase_continuity'] > 0:
            losses['phase_continuity'] = self._phase_continuity_loss(field)
        
        # Propagation loss if distance is provided
        if distance is not None and loss_weights['propagation'] > 0:
            losses['propagation'] = self._propagation_loss(field, distance)
        
        # Maxwell's equations losses if enabled
        if self.use_maxwell and loss_weights['maxwell'] > 0:
            maxwell_losses = self._maxwell_losses(field, material_properties)
            losses.update(maxwell_losses)
        
        # Quantum optics losses if enabled
        if self.use_quantum and loss_weights['quantum'] > 0:
            quantum_losses = self._quantum_losses(field)
            losses.update(quantum_losses)
        
        # Advanced physics losses if enabled
        if self.use_advanced_physics:
            advanced_losses = self._advanced_physics_losses(field)
            losses.update(advanced_losses)
        
        return losses
    
    def _get_curriculum_stage(self, epoch: int) -> Dict:
        """Get current curriculum learning stage"""
        total_epochs = 0
        for stage in self.curriculum_stages:
            total_epochs += stage['epochs']
            if epoch < total_epochs:
                return stage
        return self.curriculum_stages[-1]
    
    def _quantum_losses(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate quantum optics losses"""
        if not self.use_quantum:
            return {}
        
        losses = {}
        
        # Calculate quantum state
        quantum_state = self.quantum.calculate_state(field)
        
        # Wigner function loss
        losses['wigner'] = self._wigner_function_loss(quantum_state)
        
        # Squeezing loss
        losses['squeezing'] = self._squeezing_loss(quantum_state)
        
        # Entanglement loss
        losses['entanglement'] = self._entanglement_loss(quantum_state)
        
        return losses
    
    def _wigner_function_loss(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Calculate Wigner function loss"""
        # Calculate Wigner function
        wigner = self.quantum.calculate_wigner(quantum_state)
        
        # Wigner function should be real and normalized
        loss = torch.mean(torch.abs(torch.imag(wigner))) + \
               torch.abs(torch.sum(wigner) - 1.0)
        
        return loss
    
    def _squeezing_loss(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Calculate squeezing loss"""
        # Calculate quadrature variances
        var_x, var_p = self.quantum.calculate_quadrature_variances(quantum_state)
        
        # Heisenberg uncertainty principle: var_x * var_p >= 1/4
        loss = torch.mean(torch.relu(0.25 - var_x * var_p))
        
        return loss
    
    def _entanglement_loss(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Calculate entanglement loss"""
        # Calculate reduced density matrices
        rho_1, rho_2 = self.quantum.calculate_reduced_density_matrices(quantum_state)
        
        # Calculate von Neumann entropy
        entropy_1 = self.quantum.calculate_von_neumann_entropy(rho_1)
        entropy_2 = self.quantum.calculate_von_neumann_entropy(rho_2)
        
        # Entanglement entropy should be equal for both subsystems
        loss = torch.abs(entropy_1 - entropy_2)
        
        return loss
    
    def _helmholtz_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate Helmholtz equation loss"""
        # Calculate spatial derivatives
        dx = self.pixel_size
        dy = self.pixel_size
        
        # Second derivatives using central difference
        d2x = (field[:, :, :, 2:] - 2 * field[:, :, :, 1:-1] + field[:, :, :, :-2]) / (dx**2)
        d2y = (field[:, :, 2:, :] - 2 * field[:, :, 1:-1, :] + field[:, :, :-2, :]) / (dy**2)
        
        # Helmholtz equation: ∇²ψ + k²ψ = 0
        k = 2 * np.pi / self.wavelength
        helmholtz = d2x + d2y + (k**2) * field[:, :, 1:-1, 1:-1]
        
        return torch.mean(torch.abs(helmholtz)**2)
    
    def _energy_conservation_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate energy conservation loss"""
        # Energy should be conserved in propagation
        energy = torch.abs(field)**2
        energy_grad = torch.mean(torch.abs(energy[1:] - energy[:-1]))
        
        return energy_grad
    
    def _phase_continuity_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate phase continuity loss"""
        # Phase should be continuous
        phase = torch.angle(field)
        phase_grad_x = torch.mean(torch.abs(phase[:, :, :, 1:] - phase[:, :, :, :-1]))
        phase_grad_y = torch.mean(torch.abs(phase[:, :, 1:, :] - phase[:, :, :-1, :]))
        
        return phase_grad_x + phase_grad_y
    
    def _propagation_loss(
        self,
        field: torch.Tensor,
        distance: float
    ) -> torch.Tensor:
        """Calculate propagation loss"""
        # Propagate field using physics
        propagated = self.propagator.angular_spectrum_propagate(
            field,
            distance=distance,
            wavelength=self.wavelength,
            pixel_size=self.pixel_size
        )
        
        # Compare with network prediction
        return torch.mean(torch.abs(propagated - field)**2)
    
    def _maxwell_losses(
        self,
        field: torch.Tensor,
        material_properties: Optional[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Calculate Maxwell's equations losses"""
        if not self.use_maxwell:
            return {}
        
        # Get Maxwell's equations losses
        maxwell_losses = self.maxwell.calculate_losses(
            field,
            material_properties
        )
        
        return maxwell_losses
    
    def _advanced_physics_losses(self, field: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate advanced physics losses"""
        if not self.use_advanced_physics:
            return {}
        
        losses = {}
        
        # Poynting vector conservation
        losses['poynting'] = self._poynting_conservation_loss(field)
        
        # Angular momentum conservation
        losses['angular_momentum'] = self._angular_momentum_loss(field)
        
        # Polarization conservation
        losses['polarization'] = self._polarization_conservation_loss(field)
        
        # Wavefront curvature
        losses['wavefront'] = self._wavefront_curvature_loss(field)
        
        return losses
    
    def _poynting_conservation_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate Poynting vector conservation loss"""
        # Calculate Poynting vector components
        E = field
        H = torch.fft.fft2(E)  # Magnetic field in frequency domain
        
        # Poynting vector: S = E × H
        S_x = torch.real(E * torch.conj(H))
        S_y = torch.imag(E * torch.conj(H))
        
        # Conservation: ∇·S = 0
        div_S = (S_x[:, :, :, 1:] - S_x[:, :, :, :-1]) / self.pixel_size + \
                (S_y[:, :, 1:, :] - S_y[:, :, :-1, :]) / self.pixel_size
        
        return torch.mean(torch.abs(div_S)**2)
    
    def _angular_momentum_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate angular momentum conservation loss"""
        # Calculate angular momentum density
        r = torch.meshgrid(
            torch.linspace(-1, 1, field.shape[-2]),
            torch.linspace(-1, 1, field.shape[-1])
        )
        r = torch.stack(r, dim=0).to(field.device)
        
        # Angular momentum: L = r × p
        p = torch.abs(field)**2  # Momentum density
        L = torch.cross(r, p)
        
        # Conservation: ∂L/∂t = 0
        L_grad = torch.mean(torch.abs(L[1:] - L[:-1]))
        
        return L_grad
    
    def _polarization_conservation_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate polarization conservation loss"""
        # Calculate Stokes parameters
        I = torch.abs(field)**2
        Q = torch.real(field * torch.conj(field))
        U = torch.imag(field * torch.conj(field))
        V = torch.angle(field)
        
        # Conservation: I² = Q² + U² + V²
        conservation = I**2 - (Q**2 + U**2 + V**2)
        
        return torch.mean(torch.abs(conservation))
    
    def _wavefront_curvature_loss(self, field: torch.Tensor) -> torch.Tensor:
        """Calculate wavefront curvature loss"""
        # Calculate phase
        phase = torch.angle(field)
        
        # Calculate curvature using Laplacian
        dx = self.pixel_size
        dy = self.pixel_size
        
        d2x = (phase[:, :, :, 2:] - 2 * phase[:, :, :, 1:-1] + phase[:, :, :, :-2]) / (dx**2)
        d2y = (phase[:, :, 2:, :] - 2 * phase[:, :, 1:-1, :] + phase[:, :, :-2, :]) / (dy**2)
        
        curvature = d2x + d2y
        
        # Curvature should be smooth
        return torch.mean(torch.abs(curvature[1:] - curvature[:-1]))

class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks"""
    
    def __init__(
        self,
        model: PINN,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Optional[Dict] = None
    ):
        """
        Initialize PINN trainer
        
        Args:
            model: PINN model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            device: Device to train on
            config: Optional training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-3)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize loss weights
        self.loss_weights = {
            'helmholtz': self.config.get('helmholtz_weight', 1.0),
            'energy': self.config.get('energy_weight', 0.1),
            'phase_continuity': self.config.get('phase_continuity_weight', 0.1),
            'propagation': self.config.get('propagation_weight', 1.0),
            'maxwell': self.config.get('maxwell_weight', 1.0),
            'poynting': self.config.get('poynting_weight', 0.1),
            'angular_momentum': self.config.get('angular_momentum_weight', 0.1),
            'polarization': self.config.get('polarization_weight', 0.1),
            'wavefront': self.config.get('wavefront_weight', 0.1)
        }
        
        # Initialize training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'physics_losses': {
                'helmholtz': [],
                'energy': [],
                'phase_continuity': [],
                'propagation': [],
                'maxwell': [],
                'poynting': [],
                'angular_momentum': [],
                'polarization': [],
                'wavefront': []
            }
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Get data
            field = batch['field'].to(self.device)
            distance = batch.get('distance', None)
            if distance is not None:
                distance = distance.to(self.device)
            
            material_properties = batch.get('material_properties', None)
            
            # Forward pass
            predicted_field, physics_losses = self.model(
                field,
                distance,
                material_properties
            )
            
            # Calculate total loss
            loss = 0
            for loss_name, loss_value in physics_losses.items():
                loss += self.loss_weights[loss_name] * loss_value
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update metrics
            for loss_name, loss_value in physics_losses.items():
                self.metrics['physics_losses'][loss_name].append(loss_value.item())
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate model"""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data
                field = batch['field'].to(self.device)
                distance = batch.get('distance', None)
                if distance is not None:
                    distance = distance.to(self.device)
                
                material_properties = batch.get('material_properties', None)
                
                # Forward pass
                predicted_field, physics_losses = self.model(
                    field,
                    distance,
                    material_properties
                )
                
                # Calculate total loss
                loss = 0
                for loss_name, loss_value in physics_losses.items():
                    loss += self.loss_weights[loss_name] * loss_value
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping: bool = True,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train model
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Optional path to save model
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary of training metrics
        """
        best_val_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.metrics['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.metrics['val_loss'].append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    
                    # Save best model
                    if save_path is not None:
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'metrics': self.metrics
                        }, save_path)
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            
            # Print physics losses
            for loss_name, loss_values in self.metrics['physics_losses'].items():
                if loss_values:
                    print(f"{loss_name}: {loss_values[-1]:.6f}")
        
        return self.metrics 