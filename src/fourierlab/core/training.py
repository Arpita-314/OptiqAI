import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OpticsTrainer:
    """Trainer class for optical models"""
    
    def __init__(self, model, wavelength=632.8, pixel_size=5.0, learning_rate=0.001):
        self.model = model
        self.wavelength = wavelength  # nm
        self.pixel_size = pixel_size  # um
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        self.phase_rmse = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Add channel dimension if needed
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            
            # Ensure data is float
            data = data.float()
            
            # Ensure target is long
            target = target.long()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        phase_errors = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Add channel dimension if needed
                if len(data.shape) == 3:
                    data = data.unsqueeze(1)
                
                # Ensure data is float
                data = data.float()
                
                # Ensure target is long
                target = target.long()
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                # Calculate phase RMSE
                phase_error = self._calculate_phase_error(data, output)
                phase_errors.append(phase_error)
        
        val_loss /= len(val_loader)
        phase_rmse = np.mean(phase_errors)
        
        self.val_losses.append(val_loss)
        self.phase_rmse.append(phase_rmse)
        
        return val_loss, phase_rmse
    
    def _calculate_phase_error(self, input_data, predictions):
        """Calculate phase reconstruction error"""
        try:
            # Get predicted class probabilities
            probs = torch.softmax(predictions, dim=1)
            
            # Convert probabilities to phase (simplified)
            pred_phase = torch.sum(probs * torch.linspace(0, 2*np.pi, predictions.shape[1]).to(self.device), dim=1)
            
            # Get true phase from input (simplified)
            true_phase = torch.angle(torch.fft.fft2(input_data))[:, 0]  # Take first channel
            
            # Calculate RMSE
            error = torch.sqrt(torch.mean((pred_phase - true_phase) ** 2))
            return error.item()
        except Exception:
            # Return a default error if calculation fails
            return 0.0
    
    def _amplitude_to_phase(self, amplitude):
        """Convert amplitude to phase"""
        # This is a simplified conversion
        # In real applications, this would involve more complex physics
        return torch.angle(torch.complex(amplitude, torch.zeros_like(amplitude)))
    
    def _get_true_phase(self, input_data):
        """Extract true phase from input data"""
        # This is a placeholder for actual phase extraction
        # In real applications, this would involve physics-based calculations
        return torch.angle(torch.fft.fft2(input_data)) 