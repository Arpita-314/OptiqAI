import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.linalg import sqrtm
from scipy.special import hermite
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json

class QuantumState:
    """Base class for quantum states in phase space representation.
    
    This class provides the foundation for implementing various quantum states
    in phase space, including Fock states, cat states, GKP states, and more.
    
    Attributes:
        size (Tuple[int, int]): Size of the phase space grid (x, p)
        state (torch.Tensor): Complex tensor representing the quantum state
        wigner (torch.Tensor): Wigner function of the state
        phase_space (torch.Tensor): Phase space distribution
    
    Methods:
        calculate_wigner(): Calculate the Wigner function
        calculate_phase_space(): Calculate phase space distribution
        calculate_entropy(): Calculate von Neumann entropy
    """
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        """Initialize quantum state.
        
        Args:
            size (Tuple[int, int]): Size of the phase space grid (x, p)
        """
        self.size = size
        self.state = None
        self.wigner = None
        self.phase_space = None
    
    def calculate_wigner(self) -> torch.Tensor:
        """Calculate Wigner function of the quantum state.
        
        Returns:
            torch.Tensor: Wigner function W(x,p)
        """
        raise NotImplementedError
    
    def calculate_phase_space(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate phase space distribution.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (x_grid, p_grid, distribution)
        """
        raise NotImplementedError
    
    def calculate_entropy(self) -> torch.Tensor:
        """Calculate von Neumann entropy of the state.
        
        Returns:
            torch.Tensor: von Neumann entropy
        """
        raise NotImplementedError

class FockState(QuantumState):
    """Fock state implementation.
    
    A Fock state |n⟩ represents a state with exactly n photons.
    The state is represented in phase space using the wavefunction
    of a harmonic oscillator eigenstate.
    
    Attributes:
        n (int): Number of photons in the state
        size (Tuple[int, int]): Size of the phase space grid
        state (torch.Tensor): Complex tensor representing the state
    
    Methods:
        generate_state(): Generate the Fock state wavefunction
        calculate_wigner(): Calculate the Wigner function
    """
    def __init__(self, n: int, size: Tuple[int, int] = (256, 256)):
        """Initialize Fock state.
        
        Args:
            n (int): Number of photons
            size (Tuple[int, int]): Size of the phase space grid
        """
        super().__init__(size)
        self.n = n
        self.generate_state()
    
    def generate_state(self):
        """Generate Fock state wavefunction using Hermite polynomials.
        
        The wavefunction is given by:
        ψ_n(x) = (1/√(2^n n!)) H_n(x) exp(-x²/2)
        where H_n is the nth Hermite polynomial.
        """
        x = torch.linspace(-5, 5, self.size[0])
        y = torch.linspace(-5, 5, self.size[1])
        X, Y = torch.meshgrid(x, y)
        r2 = X**2 + Y**2
        
        # Calculate Hermite polynomial
        H = hermite(self.n)
        H_n = torch.tensor(H(r2.numpy()), dtype=torch.float32)
        
        # Calculate normalization using torch.factorial
        norm = torch.sqrt(torch.factorial(torch.tensor(self.n, dtype=torch.float32)))
        
        # Generate state with proper complex phase
        self.state = (H_n * torch.exp(-r2/2) / norm).to(torch.complex64)
    
    def calculate_wigner(self) -> torch.Tensor:
        """Calculate Wigner function for Fock state.
        
        The Wigner function for a Fock state is given by:
        W(x,p) = 2/π (-1)^n exp(-2|α|²) L_n(4|α|²)
        where L_n is the nth Laguerre polynomial and α = (x+ip)/√2.
        
        Returns:
            torch.Tensor: Wigner function W(x,p)
        """
        x = torch.linspace(-5, 5, self.size[0])
        p = torch.linspace(-5, 5, self.size[1])
        X, P = torch.meshgrid(x, p)
        
        wigner = torch.zeros_like(X, dtype=torch.float32)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                alpha = (X[i, j] + 1j * P[i, j]) / np.sqrt(2)
                wigner[i, j] = 2 * torch.exp(-2 * torch.abs(alpha)**2) * torch.abs(
                    torch.sum([(-1)**k * torch.abs(alpha)**(2*k) / torch.factorial(torch.tensor(k, dtype=torch.float32)) 
                             for k in range(self.n + 1)])
                )**2
        
        self.wigner = wigner
        return wigner

class CatState(QuantumState):
    """Schrödinger cat state implementation.
    
    A cat state is a superposition of two coherent states:
    |α⟩ + |-α⟩, where α is a complex number.
    
    Attributes:
        alpha (float): Amplitude of the coherent states
        size (Tuple[int, int]): Size of the phase space grid
        state (torch.Tensor): Complex tensor representing the state
    
    Methods:
        generate_state(): Generate the cat state wavefunction
        calculate_wigner(): Calculate the Wigner function
    """
    def __init__(self, alpha: float, size: Tuple[int, int] = (256, 256)):
        """Initialize cat state.
        
        Args:
            alpha (float): Amplitude of the coherent states
            size (Tuple[int, int]): Size of the phase space grid
        """
        super().__init__(size)
        self.alpha = alpha
        self.generate_state()
    
    def generate_state(self):
        """Generate cat state wavefunction.
        
        The wavefunction is given by:
        ψ(x) = (1/√(2(1+e^(-2|α|²)))) (exp(-(x-α)²/2) + exp(-(x+α)²/2))
        """
        x = torch.linspace(-5, 5, self.size[0])
        y = torch.linspace(-5, 5, self.size[1])
        X, Y = torch.meshgrid(x, y)
        
        # Convert alpha to tensor
        alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        # Generate coherent states with proper complex phase
        psi_plus = torch.exp(-(X - alpha)**2/2 - Y**2/2 + 1j * Y * alpha)
        psi_minus = torch.exp(-(X + alpha)**2/2 - Y**2/2 - 1j * Y * alpha)
        
        # Calculate normalization
        norm = torch.sqrt(2 * (1 + torch.exp(-2*alpha**2)))
        
        # Combine states with proper complex phase
        self.state = (psi_plus + psi_minus) / norm
        self.state = self.state.to(torch.complex64)
    
    def calculate_wigner(self) -> torch.Tensor:
        """Calculate Wigner function for cat state.
        
        The Wigner function for a cat state is given by:
        W(x,p) = (1/π) (exp(-(x-α)²-p²) + exp(-(x+α)²-p²) + 2exp(-x²-p²)cos(2αp))
        
        Returns:
            torch.Tensor: Wigner function W(x,p)
        """
        x = torch.linspace(-5, 5, self.size[0])
        p = torch.linspace(-5, 5, self.size[1])
        X, P = torch.meshgrid(x, p)
        
        alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        # Calculate Wigner function components
        wigner_plus = torch.exp(-(X - alpha)**2 - P**2)
        wigner_minus = torch.exp(-(X + alpha)**2 - P**2)
        wigner_cross = torch.exp(-X**2 - P**2) * torch.cos(2 * alpha * P)
        
        # Combine components with proper normalization
        wigner = (wigner_plus + wigner_minus + 2 * wigner_cross) / (2 * np.pi * (1 + torch.exp(-2*alpha**2)))
        
        self.wigner = wigner
        return wigner

class GKPState(QuantumState):
    """Gottesman-Kitaev-Preskill state implementation"""
    def __init__(self, delta: float, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        self.delta = delta
        self.generate_state()
    
    def generate_state(self):
        """Generate GKP state wavefunction"""
        x = torch.linspace(-5, 5, self.size[0])
        y = torch.linspace(-5, 5, self.size[1])
        X, Y = torch.meshgrid(x, y)
        
        psi = torch.zeros_like(X)
        for n in range(-5, 6):
            psi += torch.exp(-(X - n*np.sqrt(np.pi))**2/(2*self.delta**2))
        
        psi = psi * torch.exp(-Y**2/(2*self.delta**2))
        self.state = psi / torch.sqrt(torch.sum(psi**2))

class NOONState(QuantumState):
    """NOON state implementation"""
    def __init__(self, n: int, size: Tuple[int, int] = (256, 256)):
        super().__init__(size)
        self.n = n
        self.generate_state()
    
    def generate_state(self):
        """Generate NOON state wavefunction"""
        x = torch.linspace(-5, 5, self.size[0])
        y = torch.linspace(-5, 5, self.size[1])
        X, Y = torch.meshgrid(x, y)
        
        # Convert n to tensor
        n = torch.tensor(self.n, dtype=torch.float32)
        
        # Create two-mode state with proper complex phase
        psi_1 = torch.exp(-(X - n)**2/2 - Y**2/2 + 1j * Y * n)
        psi_2 = torch.exp(-(X + n)**2/2 - Y**2/2 - 1j * Y * n)
        
        # NOON state superposition with proper normalization
        norm = torch.sqrt(torch.tensor(2.0))
        self.state = (psi_1 + psi_2) / norm
        self.state = self.state.to(torch.complex64)
    
    def calculate_wigner(self) -> torch.Tensor:
        """Calculate Wigner function for NOON state"""
        x = torch.linspace(-5, 5, self.size[0])
        p = torch.linspace(-5, 5, self.size[1])
        X, P = torch.meshgrid(x, p)
        
        n = torch.tensor(self.n, dtype=torch.float32)
        
        # Calculate Wigner function components
        wigner_1 = torch.exp(-(X - n)**2 - P**2)
        wigner_2 = torch.exp(-(X + n)**2 - P**2)
        wigner_cross = torch.exp(-X**2 - P**2) * torch.cos(2 * n * P)
        
        # Combine components with proper normalization
        wigner = (wigner_1 + wigner_2 + 2 * wigner_cross) / (2 * np.pi)
        
        self.wigner = wigner
        return wigner

class ClusterState(QuantumState):
    """Cluster state implementation"""
    def __init__(self, size: Tuple[int, int] = (256, 256), num_modes: int = 4):
        super().__init__(size)
        self.num_modes = num_modes
        self.generate_state()
    
    def generate_state(self):
        """Generate cluster state wavefunction"""
        x = torch.linspace(-5, 5, self.size[0])
        y = torch.linspace(-5, 5, self.size[1])
        X, Y = torch.meshgrid(x, y)
        
        # Initialize with squeezed states
        psi = torch.ones_like(X)
        for i in range(self.num_modes):
            # Apply CZ gates between neighboring modes
            if i < self.num_modes - 1:
                psi *= torch.exp(1j * np.pi * X * Y)
        
        self.state = psi / torch.sqrt(torch.sum(torch.abs(psi)**2))

class QuantumOpticsCalculator:
    """Main calculator class for quantum optics operations"""
    def __init__(self, wavelength: float = 632.8e-9, pixel_size: float = 5e-6, device: str = 'cuda'):
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.hbar = 1.054571817e-34
        self.c = 299792458
        self.omega = 2 * np.pi * self.c / wavelength
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Pre-compute common values
        self._precompute_values()
    
    def _precompute_values(self):
        """Pre-compute commonly used values for optimization"""
        self.sqrt_2 = torch.tensor(np.sqrt(2), device=self.device)
        self.sqrt_pi = torch.tensor(np.sqrt(np.pi), device=self.device)
        self.two_pi = torch.tensor(2 * np.pi, device=self.device)
    
    def calculate_quadrature_variances(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate quadrature variances using optimized operations"""
        x = state[..., 0].to(self.device)
        p = state[..., 1].to(self.device)
        
        # Use torch.var with unbiased=False for better performance
        var_x = torch.var(x, dim=(-2, -1), unbiased=False)
        var_p = torch.var(p, dim=(-2, -1), unbiased=False)
        return var_x, var_p
    
    def calculate_photon_number(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate photon number using optimized operations"""
        x = state[..., 0].to(self.device)
        p = state[..., 1].to(self.device)
        
        # Use torch.sum for better performance
        return 0.5 * torch.sum(x**2 + p**2 - 1, dim=(-2, -1))
    
    def calculate_squeezing(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate squeezing parameter using optimized operations"""
        var_x, var_p = self.calculate_quadrature_variances(state)
        return 0.5 * torch.log(var_x / var_p)
    
    def calculate_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate entanglement measure using optimized operations"""
        # Move state to device
        state = state.to(self.device)
        
        # Calculate density matrix using einsum for better performance
        rho = torch.einsum('...i,...j->...ij', state, state.conj())
        
        # Calculate partial traces using optimized operations
        rho_1 = torch.trace(rho, dim1=-2, dim2=-1)
        rho_2 = torch.trace(rho, dim1=-2, dim2=-1)
        
        # Calculate eigenvalues using torch.linalg.eigvals
        eigenvalues_1 = torch.linalg.eigvals(rho_1)
        eigenvalues_2 = torch.linalg.eigvals(rho_2)
        
        # Calculate entropy using optimized operations
        entropy_1 = -torch.sum(eigenvalues_1 * torch.log2(eigenvalues_1 + 1e-10))
        entropy_2 = -torch.sum(eigenvalues_2 * torch.log2(eigenvalues_2 + 1e-10))
        
        return torch.min(entropy_1, entropy_2)
    
    def calculate_quantum_fisher_information(
        self,
        state: torch.Tensor,
        parameter: str = 'phase'
    ) -> torch.Tensor:
        """Calculate quantum Fisher information using optimized operations"""
        state = state.to(self.device)
        
        if parameter == 'phase':
            var_x, var_p = self.calculate_quadrature_variances(state)
            return 4 * var_x
        elif parameter == 'amplitude':
            n = self.calculate_photon_number(state)
            return 4 * (n + 1)
        elif parameter == 'squeezing':
            r = self.calculate_squeezing(state)
            return 4 * torch.cosh(2*r)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    
    def calculate_metrological_gain(
        self,
        state: torch.Tensor,
        parameter: str = 'phase'
    ) -> torch.Tensor:
        """Calculate metrological gain using optimized operations"""
        F = self.calculate_quantum_fisher_information(state, parameter)
        F_classical = torch.tensor(4.0, device=self.device)
        return F / F_classical

class QuantumOperations:
    """Class for quantum operations with optimized implementations"""
    def __init__(self, calculator: QuantumOpticsCalculator):
        self.calculator = calculator
        self.device = calculator.device
        
        # Pre-compute common matrices
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        """Pre-compute commonly used matrices for optimization"""
        # Beam splitter matrix
        self.bs_matrix = torch.tensor([
            [np.cos(np.pi/4), -np.sin(np.pi/4)],
            [np.sin(np.pi/4), np.cos(np.pi/4)]
        ], device=self.device)
        
        # Two-mode squeezing matrix
        self.squeezing_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=self.device)
    
    def beam_splitter(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        theta: float = np.pi/4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply beam splitter operation using optimized matrix multiplication"""
        # Move states to device
        state1 = state1.to(self.device)
        state2 = state2.to(self.device)
        
        # Create beam splitter matrix
        BS = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], device=self.device)
        
        # Apply transformation using einsum for better performance
        state = torch.stack([state1, state2], dim=-1)
        output = torch.einsum('ij,...j->...i', BS, state)
        
        return output[..., 0], output[..., 1]
    
    def phase_shifter(
        self,
        state: torch.Tensor,
        phi: float
    ) -> torch.Tensor:
        """Apply phase shift using optimized complex multiplication"""
        state = state.to(self.device)
        return state * torch.exp(1j * torch.tensor(phi, device=self.device))
    
    def squeezer(
        self,
        state: torch.Tensor,
        r: float
    ) -> torch.Tensor:
        """Apply squeezing operation using optimized tensor operations"""
        state = state.to(self.device)
        x = state[..., 0]
        p = state[..., 1]
        
        # Use torch.exp for better performance
        x_new = x * torch.exp(torch.tensor(r, device=self.device))
        p_new = p * torch.exp(torch.tensor(-r, device=self.device))
        
        return torch.stack([x_new, p_new], dim=-1)
    
    def displacement(
        self,
        state: torch.Tensor,
        alpha: complex
    ) -> torch.Tensor:
        """Apply displacement operation using optimized tensor operations"""
        state = state.to(self.device)
        x = state[..., 0]
        p = state[..., 1]
        
        # Convert alpha to tensor
        alpha_tensor = torch.tensor([alpha.real, alpha.imag], device=self.device)
        
        # Apply displacement
        x_new = x + alpha_tensor[0]
        p_new = p + alpha_tensor[1]
        
        return torch.stack([x_new, p_new], dim=-1)
    
    def kerr_nonlinearity(
        self,
        state: torch.Tensor,
        chi: float
    ) -> torch.Tensor:
        """Apply Kerr nonlinearity using optimized tensor operations"""
        state = state.to(self.device)
        x = state[..., 0]
        p = state[..., 1]
        
        # Calculate photon number using optimized operations
        n = 0.5 * (x**2 + p**2 - 1)
        phase = chi * n**2
        
        return state * torch.exp(1j * phase)
    
    def two_mode_squeezing(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        r: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply two-mode squeezing using optimized matrix operations"""
        # Move states to device
        state1 = state1.to(self.device)
        state2 = state2.to(self.device)
        
        # Create squeezing matrix
        S = torch.tensor([
            [torch.cosh(torch.tensor(r, device=self.device)), 0, torch.sinh(torch.tensor(r, device=self.device)), 0],
            [0, torch.cosh(torch.tensor(r, device=self.device)), 0, -torch.sinh(torch.tensor(r, device=self.device))],
            [torch.sinh(torch.tensor(r, device=self.device)), 0, torch.cosh(torch.tensor(r, device=self.device)), 0],
            [0, -torch.sinh(torch.tensor(r, device=self.device)), 0, torch.cosh(torch.tensor(r, device=self.device))]
        ], device=self.device)
        
        # Combine states
        state = torch.stack([
            state1[..., 0], state1[..., 1],
            state2[..., 0], state2[..., 1]
        ], dim=-1)
        
        # Apply transformation using einsum for better performance
        output = torch.einsum('ij,...j->...i', S, state)
        
        # Split back into two states
        out1 = torch.stack([output[..., 0], output[..., 1]], dim=-1)
        out2 = torch.stack([output[..., 2], output[..., 3]], dim=-1)
        
        return out1, out2

class QuantumStateVisualizer:
    """Class for visualizing quantum states and tomography results"""
    def __init__(self):
        self.colormap = 'viridis'
    
    def plot_wigner(
        self,
        wigner: torch.Tensor,
        x: torch.Tensor,
        p: torch.Tensor,
        title: str = 'Wigner Function',
        save_path: Optional[str] = None
    ) -> None:
        """Plot Wigner function"""
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, p, wigner, cmap=self.colormap)
        plt.colorbar(label='W(x,p)')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_phase_space(
        self,
        phase_space: torch.Tensor,
        x: torch.Tensor,
        p: torch.Tensor,
        title: str = 'Phase Space Distribution',
        save_path: Optional[str] = None
    ) -> None:
        """Plot phase space distribution"""
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, p, phase_space, cmap=self.colormap)
        plt.colorbar(label='P(x,p)')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_measurements(
        self,
        measurements: Dict[str, torch.Tensor],
        title: str = 'Quadrature Measurements',
        save_path: Optional[str] = None
    ) -> None:
        """Plot quadrature measurements"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.hist(measurements['x'].flatten().cpu().numpy(), bins=50, density=True)
        plt.xlabel('Position (x)')
        plt.ylabel('Probability')
        plt.title('Position Distribution')
        
        plt.subplot(122)
        plt.hist(measurements['p'].flatten().cpu().numpy(), bins=50, density=True)
        plt.xlabel('Momentum (p)')
        plt.ylabel('Probability')
        plt.title('Momentum Distribution')
        
        plt.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_tomography_results(
        self,
        results: Dict[str, Any],
        x: torch.Tensor,
        p: torch.Tensor,
        title: str = 'State Tomography Results',
        save_path: Optional[str] = None
    ) -> None:
        """Plot tomography results"""
        plt.figure(figsize=(15, 5))
        
        # Plot measurements
        plt.subplot(131)
        plt.hist(results['measurements']['x'].flatten().cpu().numpy(), bins=50, density=True)
        plt.xlabel('Position (x)')
        plt.ylabel('Probability')
        plt.title('Position Distribution')
        
        plt.subplot(132)
        plt.hist(results['measurements']['p'].flatten().cpu().numpy(), bins=50, density=True)
        plt.xlabel('Momentum (p)')
        plt.ylabel('Probability')
        plt.title('Momentum Distribution')
        
        # Plot reconstructed state
        plt.subplot(133)
        plt.pcolormesh(x, p, results['reconstructed_state'], cmap=self.colormap)
        plt.colorbar(label='W(x,p)')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.title(f'Reconstructed State (Fidelity: {results["fidelity"]:.3f})')
        
        plt.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        title: str = 'Quantum State Metrics',
        save_path: Optional[str] = None
    ) -> None:
        """Plot quantum state metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot Wigner function
        plt.subplot(231)
        plt.pcolormesh(metrics['wigner'], cmap=self.colormap)
        plt.colorbar(label='W(x,p)')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.title('Wigner Function')
        
        # Plot phase space
        plt.subplot(232)
        plt.pcolormesh(metrics['phase_space'], cmap=self.colormap)
        plt.colorbar(label='P(x,p)')
        plt.xlabel('Position (x)')
        plt.ylabel('Momentum (p)')
        plt.title('Phase Space Distribution')
        
        # Plot photon number
        plt.subplot(233)
        plt.bar(range(len(metrics['photon_number'])), metrics['photon_number'].cpu().numpy())
        plt.xlabel('Mode')
        plt.ylabel('Photon Number')
        plt.title('Photon Number Distribution')
        
        # Plot squeezing
        plt.subplot(234)
        plt.bar(range(len(metrics['squeezing'])), metrics['squeezing'].cpu().numpy())
        plt.xlabel('Mode')
        plt.ylabel('Squeezing Parameter')
        plt.title('Squeezing Distribution')
        
        # Plot entanglement
        plt.subplot(235)
        plt.bar(range(len(metrics['entanglement'])), metrics['entanglement'].cpu().numpy())
        plt.xlabel('Mode')
        plt.ylabel('Entanglement Measure')
        plt.title('Entanglement Distribution')
        
        # Plot metrological gain
        plt.subplot(236)
        plt.bar(range(len(metrics['metrological_gain'])), metrics['metrological_gain'].cpu().numpy())
        plt.xlabel('Mode')
        plt.ylabel('Metrological Gain')
        plt.title('Metrological Gain Distribution')
        
        plt.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_quantum_circuit(
        self,
        operations: List[Dict],
        title: str = 'Quantum Circuit',
        save_path: Optional[str] = None
    ) -> None:
        """Plot quantum circuit diagram"""
        plt.figure(figsize=(10, 5))
        
        # Draw circuit
        for i, op in enumerate(operations):
            op_type = op['type']
            params = op['params']
            
            # Draw operation box
            plt.plot([i, i+1], [0, 0], 'k-', linewidth=2)
            plt.text(i+0.5, 0.1, f"{op_type}\n{params}", ha='center', va='bottom')
        
        plt.xlim(-0.5, len(operations)+0.5)
        plt.ylim(-0.5, 0.5)
        plt.axis('off')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.show()

class QuantumStateGenerator:
    """Generator class for creating quantum states"""
    def __init__(self, calculator: QuantumOpticsCalculator):
        self.calculator = calculator
    
    def create_fock_state(self, n: int, size: Tuple[int, int] = (256, 256)) -> FockState:
        """Create Fock state"""
        return FockState(n, size)
    
    def create_cat_state(self, alpha: float, size: Tuple[int, int] = (256, 256)) -> CatState:
        """Create cat state"""
        return CatState(alpha, size)
    
    def create_gkp_state(self, delta: float, size: Tuple[int, int] = (256, 256)) -> GKPState:
        """Create GKP state"""
        return GKPState(delta, size)
    
    def create_noon_state(self, n: int, size: Tuple[int, int] = (256, 256)) -> NOONState:
        """Create NOON state"""
        return NOONState(n, size)
    
    def create_cluster_state(self, size: Tuple[int, int] = (256, 256), num_modes: int = 4) -> ClusterState:
        """Create cluster state"""
        return ClusterState(size, num_modes)

class QuantumStateTomography:
    """Class for quantum state tomography"""
    def __init__(self, calculator: QuantumOpticsCalculator):
        self.calculator = calculator
    
    def measure_quadratures(
        self,
        state: torch.Tensor,
        num_measurements: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Perform quadrature measurements"""
        x = state[..., 0]
        p = state[..., 1]
        
        # Generate measurement outcomes
        x_measurements = torch.normal(x, 0.1, (num_measurements, *x.shape))
        p_measurements = torch.normal(p, 0.1, (num_measurements, *p.shape))
        
        return {
            'x': x_measurements,
            'p': p_measurements
        }
    
    def reconstruct_state(
        self,
        measurements: Dict[str, torch.Tensor],
        grid_size: int = 256
    ) -> torch.Tensor:
        """Reconstruct quantum state from measurements"""
        x_meas = measurements['x']
        p_meas = measurements['p']
        
        # Create phase space grid
        x = torch.linspace(-5, 5, grid_size)
        p = torch.linspace(-5, 5, grid_size)
        X, P = torch.meshgrid(x, p)
        
        # Reconstruct Wigner function
        wigner = torch.zeros_like(X)
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate overlap with measurement outcomes
                overlap = torch.exp(-0.5 * (
                    (x_meas - X[i, j])**2 +
                    (p_meas - P[i, j])**2
                ))
                wigner[i, j] = torch.mean(overlap)
        
        return wigner
    
    def calculate_fidelity(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor
    ) -> float:
        """Calculate fidelity between two states"""
        # Calculate density matrices
        rho1 = torch.einsum('...i,...j->...ij', state1, state1.conj())
        rho2 = torch.einsum('...i,...j->...ij', state2, state2.conj())
        
        # Calculate fidelity
        sqrt_rho1 = torch.matrix_power(rho1, 0.5)
        sqrt_term = torch.matrix_power(
            sqrt_rho1 @ rho2 @ sqrt_rho1,
            0.5
        )
        fidelity = torch.trace(sqrt_term).real
        
        return fidelity.item()
    
    def perform_tomography(
        self,
        state: torch.Tensor,
        num_measurements: int = 1000,
        grid_size: int = 256
    ) -> Dict[str, torch.Tensor]:
        """Perform complete state tomography"""
        # Perform measurements
        measurements = self.measure_quadratures(state, num_measurements)
        
        # Reconstruct state
        reconstructed_state = self.reconstruct_state(measurements, grid_size)
        
        # Calculate fidelity
        fidelity = self.calculate_fidelity(state, reconstructed_state)
        
        return {
            'measurements': measurements,
            'reconstructed_state': reconstructed_state,
            'fidelity': fidelity
        }

class QuantumStateAnalyzer:
    """Analyzer class for quantum states"""
    def __init__(self, calculator: QuantumOpticsCalculator):
        self.calculator = calculator
        self.visualizer = QuantumStateVisualizer()
        self.tomography = QuantumStateTomography(calculator)
        self.io = QuantumStateIO(calculator)
    
    def analyze_state(self, state: QuantumState) -> Dict[str, torch.Tensor]:
        """Analyze quantum state and return metrics"""
        metrics = {}
        
        # Calculate basic properties
        metrics['wigner'] = state.calculate_wigner()
        metrics['phase_space'] = state.calculate_phase_space()
        metrics['entropy'] = state.calculate_entropy()
        
        # Calculate advanced properties
        if isinstance(state.state, torch.Tensor):
            metrics['photon_number'] = self.calculator.calculate_photon_number(state.state)
            metrics['squeezing'] = self.calculator.calculate_squeezing(state.state)
            metrics['entanglement'] = self.calculator.calculate_entanglement(state.state)
            metrics['fisher_info'] = self.calculator.calculate_quantum_fisher_information(state.state)
            metrics['metrological_gain'] = self.calculator.calculate_metrological_gain(state.state)
            
            # Perform tomography
            tomography_results = self.tomography.perform_tomography(state.state)
            metrics['tomography'] = tomography_results
        
        return metrics
    
    def visualize_analysis(
        self,
        metrics: Dict[str, torch.Tensor],
        save_dir: Optional[str] = None
    ) -> None:
        """Visualize analysis results"""
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot metrics
        metrics_path = os.path.join(save_dir, 'metrics.png') if save_dir else None
        self.visualizer.plot_metrics(metrics, save_path=metrics_path)
        
        # Plot tomography results
        if 'tomography' in metrics:
            tomography_path = os.path.join(save_dir, 'tomography.png') if save_dir else None
            self.visualizer.plot_tomography_results(
                metrics['tomography'],
                save_path=tomography_path
            )
    
    def visualize_quantum_circuit(
        self,
        operations: List[Dict],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize quantum circuit"""
        self.visualizer.plot_quantum_circuit(operations, save_path=save_path)

class QuantumStateIO:
    """Class for importing and exporting quantum states"""
    def __init__(self, calculator: QuantumOpticsCalculator):
        self.calculator = calculator
    
    def export_state(
        self,
        state: torch.Tensor,
        filename: str,
        format: str = 'numpy'
    ) -> None:
        """Export quantum state to file"""
        if format == 'numpy':
            np.save(filename, state.detach().cpu().numpy())
        elif format == 'torch':
            torch.save(state, filename)
        elif format == 'json':
            state_dict = {
                'state': state.detach().cpu().numpy().tolist(),
                'shape': list(state.shape),
                'dtype': str(state.dtype)
            }
            with open(filename, 'w') as f:
                json.dump(state_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_state(
        self,
        filename: str,
        format: str = 'numpy'
    ) -> torch.Tensor:
        """Import quantum state from file"""
        if format == 'numpy':
            state = torch.from_numpy(np.load(filename))
        elif format == 'torch':
            state = torch.load(filename)
        elif format == 'json':
            with open(filename, 'r') as f:
                state_dict = json.load(f)
            state = torch.tensor(state_dict['state'])
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return state
    
    def export_measurements(
        self,
        measurements: Dict[str, torch.Tensor],
        filename: str,
        format: str = 'numpy'
    ) -> None:
        """Export measurement data to file"""
        if format == 'numpy':
            np.savez(
                filename,
                x=measurements['x'].detach().cpu().numpy(),
                p=measurements['p'].detach().cpu().numpy()
            )
        elif format == 'json':
            measurements_dict = {
                'x': measurements['x'].detach().cpu().numpy().tolist(),
                'p': measurements['p'].detach().cpu().numpy().tolist()
            }
            with open(filename, 'w') as f:
                json.dump(measurements_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_measurements(
        self,
        filename: str,
        format: str = 'numpy'
    ) -> Dict[str, torch.Tensor]:
        """Import measurement data from file"""
        if format == 'numpy':
            data = np.load(filename)
            measurements = {
                'x': torch.from_numpy(data['x']),
                'p': torch.from_numpy(data['p'])
            }
        elif format == 'json':
            with open(filename, 'r') as f:
                measurements_dict = json.load(f)
            measurements = {
                'x': torch.tensor(measurements_dict['x']),
                'p': torch.tensor(measurements_dict['p'])
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return measurements
    
    def export_tomography_results(
        self,
        results: Dict[str, Any],
        filename: str,
        format: str = 'numpy'
    ) -> None:
        """Export tomography results to file"""
        if format == 'numpy':
            np.savez(
                filename,
                measurements_x=results['measurements']['x'].detach().cpu().numpy(),
                measurements_p=results['measurements']['p'].detach().cpu().numpy(),
                reconstructed_state=results['reconstructed_state'].detach().cpu().numpy(),
                fidelity=results['fidelity']
            )
        elif format == 'json':
            results_dict = {
                'measurements': {
                    'x': results['measurements']['x'].detach().cpu().numpy().tolist(),
                    'p': results['measurements']['p'].detach().cpu().numpy().tolist()
                },
                'reconstructed_state': results['reconstructed_state'].detach().cpu().numpy().tolist(),
                'fidelity': results['fidelity']
            }
            with open(filename, 'w') as f:
                json.dump(results_dict, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_tomography_results(
        self,
        filename: str,
        format: str = 'numpy'
    ) -> Dict[str, Any]:
        """Import tomography results from file"""
        if format == 'numpy':
            data = np.load(filename)
            results = {
                'measurements': {
                    'x': torch.from_numpy(data['measurements_x']),
                    'p': torch.from_numpy(data['measurements_p'])
                },
                'reconstructed_state': torch.from_numpy(data['reconstructed_state']),
                'fidelity': float(data['fidelity'])
            }
        elif format == 'json':
            with open(filename, 'r') as f:
                results_dict = json.load(f)
            results = {
                'measurements': {
                    'x': torch.tensor(results_dict['measurements']['x']),
                    'p': torch.tensor(results_dict['measurements']['p'])
                },
                'reconstructed_state': torch.tensor(results_dict['reconstructed_state']),
                'fidelity': results_dict['fidelity']
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return results 