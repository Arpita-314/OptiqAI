import torch
import numpy as np
from typing import Tuple, Optional
from .quantum_optics_calculator import QuantumOpticsCalculator

class QuantumOperations:
    """Implementation of quantum optical operations"""
    
    def __init__(self, calculator: Optional[QuantumOpticsCalculator] = None):
        self.calculator = calculator or QuantumOpticsCalculator()
        self.device = self.calculator.device
    
    def beam_splitter(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        theta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a beam splitter operation"""
        U = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]
        ], dtype=torch.complex64, device=self.device)
        
        out1 = U[0,0] * state1 + U[0,1] * state2
        out2 = U[1,0] * state1 + U[1,1] * state2
        
        return out1, out2
    
    def phase_shifter(self, state: torch.Tensor, phi: float) -> torch.Tensor:
        """Apply a phase shift"""
        return torch.exp(1j * phi) * state
    
    def squeezer(self, state: torch.Tensor, r: float) -> torch.Tensor:
        """Apply single-mode squeezing"""
        n = torch.arange(len(state), device=self.device)
        S = torch.exp(-r/2) * torch.sum(
            torch.sqrt(torch.factorial(n)) / torch.factorial(n/2) *
            (torch.tanh(r)/2)**(n/2) * state
        )
        return S / torch.sqrt(torch.sum(torch.abs(S)**2))
    
    def displacement(self, state: torch.Tensor, alpha: complex) -> torch.Tensor:
        """Apply displacement operation"""
        D = self.calculator.displacement_operator(alpha)
        return D @ state
    
    def kerr_nonlinearity(self, state: torch.Tensor, chi: float) -> torch.Tensor:
        """Apply Kerr nonlinearity"""
        n = torch.arange(len(state), device=self.device)
        K = torch.exp(1j * chi * n * (n-1))
        return K * state
    
    def two_mode_squeezing(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        r: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply two-mode squeezing"""
        n = torch.arange(len(state1), device=self.device)
        S = torch.exp(-r/2) * torch.sum(
            torch.sqrt(torch.factorial(n)) / torch.factorial(n/2) *
            (torch.tanh(r)/2)**(n/2) * torch.outer(state1, state2)
        )
        S = S / torch.sqrt(torch.sum(torch.abs(S)**2))
        return S[0], S[1]
    
    def interferometer(
        self,
        states: torch.Tensor,
        unitary: torch.Tensor
    ) -> torch.Tensor:
        """Apply a general interferometer transformation"""
        return torch.matmul(unitary, states)
    
    def controlled_phase(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        phi: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a controlled phase gate"""
        CZ = torch.eye(len(state1), dtype=torch.complex64, device=self.device)
        CZ[1,1] = torch.exp(1j * phi)
        out1 = CZ @ state1
        out2 = state2
        return out1, out2 