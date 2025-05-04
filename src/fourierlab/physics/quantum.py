import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.linalg import sqrtm
from scipy.special import hermite

class QuantumOptics:
    """Quantum optics calculations for optical fields"""
    
    def __init__(
        self,
        wavelength: float = 632.8e-9,
        pixel_size: float = 5e-6,
        num_modes: int = 2
    ):
        """
        Initialize quantum optics calculator
        
        Args:
            wavelength: Wavelength of light in meters
            pixel_size: Size of each pixel in meters
            num_modes: Number of optical modes to consider
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.num_modes = num_modes
        
        # Calculate fundamental constants
        self.hbar = 1.054571817e-34  # Reduced Planck constant
        self.c = 299792458  # Speed of light
        self.omega = 2 * np.pi * self.c / wavelength  # Angular frequency
    
    def calculate_state(
        self,
        field: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate quantum state from classical field
        
        Args:
            field: Complex optical field
            
        Returns:
            Quantum state vector
        """
        # Convert field to quantum state
        amplitude = torch.abs(field)
        phase = torch.angle(field)
        
        # Calculate quadrature amplitudes
        x = amplitude * torch.cos(phase)
        p = amplitude * torch.sin(phase)
        
        # Combine into state vector
        state = torch.stack([x, p], dim=-1)
        
        return state
    
    def calculate_wigner(
        self,
        state: torch.Tensor,
        x_range: Optional[Tuple[float, float]] = None,
        p_range: Optional[Tuple[float, float]] = None,
        num_points: int = 100
    ) -> torch.Tensor:
        """
        Calculate Wigner function
        
        Args:
            state: Quantum state vector
            x_range: Range of x quadrature
            p_range: Range of p quadrature
            num_points: Number of points in each dimension
            
        Returns:
            Wigner function
        """
        if x_range is None:
            x_range = (-5, 5)
        if p_range is None:
            p_range = (-5, 5)
        
        # Create quadrature grid
        x = torch.linspace(x_range[0], x_range[1], num_points)
        p = torch.linspace(p_range[0], p_range[1], num_points)
        X, P = torch.meshgrid(x, p)
        
        # Calculate Wigner function
        wigner = torch.zeros_like(X)
        
        for i in range(num_points):
            for j in range(num_points):
                # Displacement operator
                D = torch.exp(1j * (X[i, j] * state[..., 0] + P[i, j] * state[..., 1]))
                
                # Wigner function
                wigner[i, j] = torch.mean(torch.real(D))
        
        return wigner
    
    def calculate_quadrature_variances(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate quadrature variances
        
        Args:
            state: Quantum state vector
            
        Returns:
            Tuple of (x variance, p variance)
        """
        # Extract quadratures
        x = state[..., 0]
        p = state[..., 1]
        
        # Calculate variances
        var_x = torch.var(x, dim=(-2, -1))
        var_p = torch.var(p, dim=(-2, -1))
        
        return var_x, var_p
    
    def calculate_reduced_density_matrices(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reduced density matrices for bipartite system
        
        Args:
            state: Quantum state vector
            
        Returns:
            Tuple of (reduced density matrix 1, reduced density matrix 2)
        """
        # Reshape state for bipartite system
        state = state.reshape(-1, 2, 2)
        
        # Calculate density matrix
        rho = torch.einsum('...i,...j->...ij', state, state.conj())
        
        # Calculate reduced density matrices
        rho_1 = torch.trace(rho, dim1=-2, dim2=-1)
        rho_2 = torch.trace(rho, dim1=-2, dim2=-1)
        
        return rho_1, rho_2
    
    def calculate_von_neumann_entropy(
        self,
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate von Neumann entropy
        
        Args:
            rho: Density matrix
            
        Returns:
            von Neumann entropy
        """
        # Calculate eigenvalues
        eigenvalues = torch.linalg.eigvals(rho)
        
        # Calculate entropy
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-10))
        
        return entropy
    
    def calculate_photon_number(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate photon number
        
        Args:
            state: Quantum state vector
            
        Returns:
            Photon number
        """
        # Calculate quadratures
        x = state[..., 0]
        p = state[..., 1]
        
        # Calculate photon number
        n = 0.5 * (x**2 + p**2 - 1)
        
        return n
    
    def calculate_squeezing(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate squeezing parameter
        
        Args:
            state: Quantum state vector
            
        Returns:
            Squeezing parameter
        """
        # Calculate quadrature variances
        var_x, var_p = self.calculate_quadrature_variances(state)
        
        # Calculate squeezing
        r = 0.5 * torch.log(var_x / var_p)
        
        return r
    
    def calculate_entanglement(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate entanglement measure
        
        Args:
            state: Quantum state vector
            
        Returns:
            Entanglement measure
        """
        # Calculate reduced density matrices
        rho_1, rho_2 = self.calculate_reduced_density_matrices(state)
        
        # Calculate von Neumann entropy
        entropy_1 = self.calculate_von_neumann_entropy(rho_1)
        entropy_2 = self.calculate_von_neumann_entropy(rho_2)
        
        # Calculate entanglement
        entanglement = torch.min(entropy_1, entropy_2)
        
        return entanglement
    
    def calculate_phase_space(
        self,
        state: torch.Tensor,
        num_points: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate phase space distribution
        
        Args:
            state: Quantum state vector
            num_points: Number of points in each dimension
            
        Returns:
            Tuple of (x grid, p grid, distribution)
        """
        # Create quadrature grid
        x = torch.linspace(-5, 5, num_points)
        p = torch.linspace(-5, 5, num_points)
        X, P = torch.meshgrid(x, p)
        
        # Calculate Husimi Q function
        Q = torch.zeros_like(X)
        
        for i in range(num_points):
            for j in range(num_points):
                # Coherent state
                alpha = (X[i, j] + 1j * P[i, j]) / np.sqrt(2)
                
                # Q function
                Q[i, j] = torch.exp(-torch.abs(alpha - state)**2)
        
        return X, P, Q
    
    def generate_fock_state(
        self,
        n: int,
        size: Tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """
        Generate Fock state |n⟩
        
        Args:
            n: Photon number
            size: Size of the field
            
        Returns:
            Fock state field
        """
        # Create coordinate grid
        x = torch.linspace(-5, 5, size[0])
        y = torch.linspace(-5, 5, size[1])
        X, Y = torch.meshgrid(x, y)
        r2 = X**2 + Y**2
        
        # Hermite polynomial
        H = hermite(n)
        H_n = torch.tensor(H(r2.numpy()), dtype=torch.float32)
        
        # Fock state wavefunction
        psi = H_n * torch.exp(-r2/2) / torch.sqrt(torch.tensor(np.math.factorial(n)))
        
        return psi
    
    def generate_cat_state(
        self,
        alpha: float,
        size: Tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """
        Generate Schrödinger cat state
        
        Args:
            alpha: Coherent state amplitude
            size: Size of the field
            
        Returns:
            Cat state field
        """
        # Create coordinate grid
        x = torch.linspace(-5, 5, size[0])
        y = torch.linspace(-5, 5, size[1])
        X, Y = torch.meshgrid(x, y)
        
        # Coherent states
        psi_plus = torch.exp(-(X - alpha)**2/2 - Y**2/2)
        psi_minus = torch.exp(-(X + alpha)**2/2 - Y**2/2)
        
        # Cat state
        psi = (psi_plus + psi_minus) / torch.sqrt(2 * (1 + torch.exp(-2*alpha**2)))
        
        return psi
    
    def generate_gkp_state(
        self,
        delta: float,
        size: Tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """
        Generate Gottesman-Kitaev-Preskill (GKP) state
        
        Args:
            delta: Squeezing parameter
            size: Size of the field
            
        Returns:
            GKP state field
        """
        # Create coordinate grid
        x = torch.linspace(-5, 5, size[0])
        y = torch.linspace(-5, 5, size[1])
        X, Y = torch.meshgrid(x, y)
        
        # GKP state
        psi = torch.zeros_like(X)
        for n in range(-5, 6):
            psi += torch.exp(-(X - n*np.sqrt(np.pi))**2/(2*delta**2))
        
        psi = psi * torch.exp(-Y**2/(2*delta**2))
        psi = psi / torch.sqrt(torch.sum(psi**2))
        
        return psi
    
    def calculate_negativity(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Wigner negativity
        
        Args:
            state: Quantum state vector
            
        Returns:
            Wigner negativity
        """
        # Calculate Wigner function
        wigner = self.calculate_wigner(state)
        
        # Calculate negativity
        negativity = torch.sum(torch.abs(torch.minimum(wigner, torch.zeros_like(wigner))))
        
        return negativity
    
    def calculate_phase_space_entropy(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate phase space entropy
        
        Args:
            state: Quantum state vector
            
        Returns:
            Phase space entropy
        """
        # Calculate Husimi Q function
        _, _, Q = self.calculate_phase_space(state)
        
        # Normalize Q function
        Q = Q / torch.sum(Q)
        
        # Calculate entropy
        entropy = -torch.sum(Q * torch.log2(Q + 1e-10))
        
        return entropy
    
    def calculate_quantum_fisher_information(
        self,
        state: torch.Tensor,
        parameter: str = 'phase'
    ) -> torch.Tensor:
        """
        Calculate quantum Fisher information
        
        Args:
            state: Quantum state vector
            parameter: Parameter to estimate ('phase', 'amplitude', 'squeezing')
            
        Returns:
            Quantum Fisher information
        """
        if parameter == 'phase':
            # Phase estimation
            var_x, var_p = self.calculate_quadrature_variances(state)
            F = 4 * var_x
        elif parameter == 'amplitude':
            # Amplitude estimation
            n = self.calculate_photon_number(state)
            F = 4 * (n + 1)
        elif parameter == 'squeezing':
            # Squeezing estimation
            r = self.calculate_squeezing(state)
            F = 4 * torch.cosh(2*r)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        return F
    
    def calculate_metrological_gain(
        self,
        state: torch.Tensor,
        parameter: str = 'phase'
    ) -> torch.Tensor:
        """
        Calculate metrological gain
        
        Args:
            state: Quantum state vector
            parameter: Parameter to estimate
            
        Returns:
            Metrological gain
        """
        # Calculate quantum Fisher information
        F = self.calculate_quantum_fisher_information(state, parameter)
        
        # Calculate classical Fisher information for coherent state
        F_classical = 4
        
        # Calculate gain
        gain = F / F_classical
        
        return gain
    
    def calculate_quantum_cramer_rao_bound(
        self,
        state: torch.Tensor,
        parameter: str = 'phase',
        num_measurements: int = 1000
    ) -> torch.Tensor:
        """
        Calculate quantum Cramér-Rao bound
        
        Args:
            state: Quantum state vector
            parameter: Parameter to estimate
            num_measurements: Number of measurements
            
        Returns:
            Quantum Cramér-Rao bound
        """
        # Calculate quantum Fisher information
        F = self.calculate_quantum_fisher_information(state, parameter)
        
        # Calculate bound
        bound = 1 / (F * num_measurements)
        
        return bound 