from PyQt5.QtCore import QObject, pyqtSignal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from fourierlab.ml.models import FourierCNN
from fourierlab.ml.arch_search import find_best_cnn
from fourierlab.core.training import OpticsTrainer

class TrainingManager(QObject):
    training_progress = pyqtSignal(int)  # Progress percentage
    training_complete = pyqtSignal(dict)  # Training results
    training_error = pyqtSignal(str)  # Error message
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_training(self, dataset, model_type="FourierCNN", epochs=10):
        """Setup the training configuration"""
        try:
            # Split data into train/val
            val_split = 0.2
            val_size = max(1, int(len(dataset) * val_split))
            train_size = len(dataset) - val_size
            train_data, val_data = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            batch_size = min(8, len(dataset))  # Ensure batch size isn't larger than dataset
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            # Get number of classes from the dataset
            if hasattr(dataset, 'classes'):
                num_classes = len(dataset.classes)
            else:
                # Try to infer from targets
                sample_data, sample_target = dataset[0]
                if isinstance(sample_target, (int, torch.Tensor)):
                    targets = [target for _, target in dataset]
                    num_classes = max(targets) + 1
                else:
                    raise ValueError("Could not determine number of classes from dataset")
            
            # Initialize model
            if model_type == "FourierCNN":
                self.model = FourierCNN(in_channels=1, num_classes=num_classes)
            else:  # AutoML Search
                self.model = find_best_cnn(dataset, num_classes=num_classes)
            
            self.model = self.model.to(self.device)
            
            # Initialize trainer
            self.trainer = OpticsTrainer(
                self.model,
                wavelength=632.8,  # Default wavelength in nm
                pixel_size=5.0     # Default pixel size in um
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            self.training_error.emit(f"Error setting up training: {str(e)}")
            return None, None
    
    def train(self, train_loader, val_loader, epochs):
        """Run the training process"""
        try:
            if self.trainer is None:
                raise ValueError("Trainer not initialized. Call setup_training first.")
            
            # Training loop with progress updates
            for epoch in range(epochs):
                self.trainer.train_epoch(train_loader)
                val_loss, phase_rmse = self.trainer.validate(val_loader)
                
                # Calculate and emit progress
                progress = int((epoch + 1) / epochs * 100)
                self.training_progress.emit(progress)
                
                # Store results
                results = {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "phase_rmse": phase_rmse
                }
                self.training_complete.emit(results)
            
            return True
                
        except Exception as e:
            self.training_error.emit(f"Error during training: {str(e)}")
            return False
    
    def save_model(self, path):
        """Save the trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            torch.save(self.model.state_dict(), path)
            return True
            
        except Exception as e:
            self.training_error.emit(f"Error saving model: {str(e)}")
            return False 