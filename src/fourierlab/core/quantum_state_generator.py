import torch
import numpy as np
from typing import Tuple, Optional
from .quantum_optics_calculator import QuantumOpticsCalculator

class QuantumStateGenerator:
    """Generator for various quantum states"""
    
    def __init__(self, calculator: Optional[QuantumOpticsCalculator] = None):
        self.calculator = calculator or QuantumOpticsCalculator()
        self.device = self.calculator.device
    
    def create_fock_state(self, n: int, size: Tuple[int, int]) -> torch.Tensor:
        """Create a Fock state |n⟩"""
        state = torch.zeros(size[0], dtype=torch.complex64, device=self.device)
        state[n] = 1.0
        return state
    
    def create_coherent_state(self, alpha: complex, size: Tuple[int, int]) -> torch.Tensor:
        """Create a coherent state |α⟩"""
        n = torch.arange(size[0], dtype=torch.float32, device=self.device)
        state = torch.exp(-0.5 * abs(alpha)**2) * (alpha**n) / torch.sqrt(torch.factorial(n))
        return state
    
    def create_cat_state(self, alpha: float, size: Tuple[int, int]) -> torch.Tensor:
        """Create a cat state (|α⟩ + |-α⟩)/N"""
        psi_plus = self.create_coherent_state(alpha, size)
        psi_minus = self.create_coherent_state(-alpha, size)
        N = torch.sqrt(2 * (1 + torch.exp(-2 * abs(alpha)**2)))
        return (psi_plus + psi_minus) / N
    
    def create_gkp_state(self, delta: float, size: Tuple[int, int]) -> torch.Tensor:
        """Create an approximate GKP state"""
        x = torch.linspace(-5, 5, size[0], device=self.device)
        gaussian_envelope = torch.exp(-x**2 / (2 * delta**2))
        comb = torch.sum(torch.stack([torch.exp(-(x - 2*np.pi*k)**2 / (2*delta**2)) 
                                    for k in range(-2, 3)]), dim=0)
        state = gaussian_envelope * comb
        return state / torch.sqrt(torch.sum(torch.abs(state)**2))
    
    def create_noon_state(self, n: int, size: Tuple[int, int]) -> torch.Tensor:
        """Create a NOON state (|n,0⟩ + |0,n⟩)/√2"""
        state1 = torch.zeros(size[0], dtype=torch.complex64, device=self.device)
        state2 = torch.zeros(size[0], dtype=torch.complex64, device=self.device)
        
        state1[n] = 1.0  # |n,0⟩
        state2[0] = 1.0  # |0,n⟩
        
        return (state1 + state2) / np.sqrt(2)
    
    def create_cluster_state(self, size: Tuple[int, int], num_modes: int) -> torch.Tensor:
        """Create a cluster state with num_modes"""
        # Start with all modes in |+⟩ state
        plus_state = torch.ones(size[0], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        
        # Apply CZ gates between neighboring modes
        state = plus_state
        for i in range(num_modes-1):
            CZ = torch.eye(size[0], dtype=torch.complex64, device=self.device)
            CZ[1,1] = -1
            state = CZ @ state
        
        return state
    
    def create_squeezed_vacuum(self, r: float, size: Tuple[int, int]) -> torch.Tensor:
        """Create a squeezed vacuum state"""
        n = torch.arange(0, size[0], 2, device=self.device)  # Even numbers only
        coeffs = torch.sqrt(torch.factorial(n)) / torch.factorial(n/2)
        coeffs *= torch.tanh(r)**(n/2) / torch.cosh(r)
        
        state = torch.zeros(size[0], dtype=torch.complex64, device=self.device)
        state[::2] = coeffs
        return state / torch.sqrt(torch.sum(torch.abs(state)**2))
    
    def create_thermal_state(self, n_th: float, size: Tuple[int, int]) -> torch.Tensor:
        """Create a thermal state with mean photon number n_th"""
        n = torch.arange(size[0], device=self.device)
        state = torch.sqrt((n_th/(1+n_th))**n / (1+n_th))
        return state 