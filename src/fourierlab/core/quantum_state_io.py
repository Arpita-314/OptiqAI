import torch
import numpy as np
from typing import Optional, Dict, Any
import json
from .quantum_optics_calculator import QuantumOpticsCalculator

class QuantumStateIO:
    """Input/Output operations for quantum states"""
    
    def __init__(self, calculator: Optional[QuantumOpticsCalculator] = None):
        self.calculator = calculator or QuantumOpticsCalculator()
        self.device = self.calculator.device
    
    def export_state(
        self,
        state: torch.Tensor,
        file_path: str,
        format: str = 'numpy',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Export quantum state to file"""
        # Convert state to CPU and numpy
        state_np = state.detach().cpu().numpy()
        
        if format == 'numpy':
            np.save(file_path, state_np)
        elif format == 'txt':
            np.savetxt(file_path, state_np)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = file_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
    
    def import_state(
        self,
        file_path: str,
        format: str = 'numpy'
    ) -> torch.Tensor:
        """Import quantum state from file"""
        if format == 'numpy':
            state_np = np.load(file_path)
        elif format == 'txt':
            state_np = np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Convert to torch tensor and move to device
        state = torch.from_numpy(state_np).to(self.device)
        
        # Try to load metadata
        try:
            metadata_path = file_path + '.meta.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return state, metadata
        except:
            return state
    
    def export_measurements(
        self,
        measurements: Dict[str, Any],
        file_path: str
    ) -> None:
        """Export measurement results"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_measurements = {}
        for key, value in measurements.items():
            if isinstance(value, np.ndarray):
                serializable_measurements[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_measurements[key] = value.detach().cpu().numpy().tolist()
            else:
                serializable_measurements[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(serializable_measurements, f)
    
    def import_measurements(self, file_path: str) -> Dict[str, Any]:
        """Import measurement results"""
        with open(file_path, 'r') as f:
            measurements = json.load(f)
        
        # Convert lists back to numpy arrays
        for key, value in measurements.items():
            if isinstance(value, list):
                measurements[key] = np.array(value)
        
        return measurements 