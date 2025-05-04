import os
from PyQt5.QtCore import QObject, pyqtSignal
from fourierlab.core.preprocessing import OpticalPreprocessor
from fourierlab.core.data_loader import FourierDataset

class DataManager(QObject):
    data_loaded = pyqtSignal(str)  # Signal emitted when data is loaded
    preprocessing_complete = pyqtSignal(object)  # Signal emitted when preprocessing is done
    
    def __init__(self):
        super().__init__()
        self.data_dir = None
        self.dataset = None
        self.processed_data = None
        self.preprocessor = OpticalPreprocessor()
    
    def load_data(self, directory):
        """Load data from the specified directory"""
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        try:
            self.data_dir = directory
            self.dataset = FourierDataset(directory)
            self.data_loaded.emit(f"Loaded {len(self.dataset)} samples from {directory}")
            return True
        except Exception as e:
            self.data_loaded.emit(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self, target_size=(256, 256)):
        """Preprocess the loaded data"""
        if self.dataset is None:
            raise ValueError("No data loaded. Please load data first.")
        
        try:
            self.preprocessor.target_size = target_size
            self.processed_data = self.preprocessor.run(self.dataset.image_paths)
            self.preprocessing_complete.emit(self.processed_data)
            return True
        except Exception as e:
            self.preprocessing_complete.emit(f"Error during preprocessing: {str(e)}")
            return False
    
    def get_dataset_info(self):
        """Get information about the loaded dataset"""
        if self.dataset is None:
            return "No data loaded"
        
        return {
            "num_samples": len(self.dataset),
            "classes": self.dataset.classes,
            "image_shape": self.dataset.image_shape if hasattr(self.dataset, 'image_shape') else None
        } 