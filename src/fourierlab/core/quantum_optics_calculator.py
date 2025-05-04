import torch
import numpy as np
from typing import Tuple, Dict, Any

class QuantumOpticsCalculator:
    """Core calculator for quantum optics operations"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._precompute_values()
    
    def _precompute_values(self):
        """Precompute commonly used values"""
        self.hbar = 1.0  # Natural units
        self.max_photon_number = 100
        
        # Create photon number operator matrix
        n = torch.arange(self.max_photon_number, device=self.device)
        self.n_op = torch.diag(n)
        
        # Create annihilation operator matrix
        self.a_op = torch.diag(torch.sqrt(n[1:]), 1)
        
        # Create creation operator matrix
        self.a_dag_op = torch.diag(torch.sqrt(n[1:]), -1)
    
    def calculate_photon_number(self, state: torch.Tensor) -> float:
        """Calculate average photon number"""
        return torch.abs(torch.sum(state * torch.conj(state) * torch.arange(len(state), device=self.device)))
    
    def calculate_quadrature_variances(self, state: torch.Tensor) -> Tuple[float, float]:
        """Calculate quadrature variances"""
        x = (self.a_op + self.a_dag_op) / np.sqrt(2)
        p = 1j * (self.a_dag_op - self.a_op) / np.sqrt(2)
        
        x_var = torch.var(torch.abs(x @ state))
        p_var = torch.var(torch.abs(p @ state))
        
        return x_var.item(), p_var.item()
    
    def calculate_squeezing(self, state: torch.Tensor) -> float:
        """Calculate squeezing parameter"""
        x_var, p_var = self.calculate_quadrature_variances(state)
        return -10 * torch.log10(torch.tensor(min(x_var, p_var) / 0.5))
    
    def calculate_wigner(self, state: torch.Tensor, grid_size: int = 100) -> torch.Tensor:
        """Calculate Wigner function"""
        x = torch.linspace(-5, 5, grid_size, device=self.device)
        p = torch.linspace(-5, 5, grid_size, device=self.device)
        X, P = torch.meshgrid(x, p, indexing='ij')
        
        alpha = (X + 1j*P) / np.sqrt(2)
        D = self.displacement_operator(alpha)
        rho = torch.outer(state, torch.conj(state))
        
        wigner = torch.real(torch.trace(D @ rho)) / (np.pi)
        return wigner
    
    def displacement_operator(self, alpha: torch.Tensor) -> torch.Tensor:
        """Create displacement operator"""
        D = torch.matrix_exp(alpha * self.a_dag_op - torch.conj(alpha) * self.a_op)
        return D
    
    def calculate_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """Calculate fidelity between two states"""
        return torch.abs(torch.sum(torch.conj(state1) * state2))**2
    
    def calculate_purity(self, state: torch.Tensor) -> float:
        """Calculate state purity"""
        rho = torch.outer(state, torch.conj(state))
        return torch.abs(torch.trace(rho @ rho))
    
    def calculate_entropy(self, state: torch.Tensor) -> float:
        """Calculate von Neumann entropy"""
        rho = torch.outer(state, torch.conj(state))
        eigenvalues = torch.linalg.eigvals(rho)
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-15))
        return entropy.real 