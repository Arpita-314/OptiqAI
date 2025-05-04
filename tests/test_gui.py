import sys
import os
import unittest
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import torch
from torch.utils.data import TensorDataset
import numpy as np
from PIL import Image
from fourierlab.UI.gui.main_window import MainWindow
from fourierlab.UI.gui.data_manager import DataManager
from fourierlab.UI.gui.training_manager import TrainingManager

class TestGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create QApplication instance
        cls.app = QApplication(sys.argv)
        
        # Create test data directory
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create test data subdirectories for classes
        cls.class_dirs = ["class1", "class2"]
        for class_dir in cls.class_dirs:
            os.makedirs(os.path.join(cls.test_data_dir, class_dir), exist_ok=True)
        
        # Create some dummy test images
        cls._create_test_images()
    
    @classmethod
    def tearDownClass(cls):
        # Clean up test data
        import shutil
        shutil.rmtree(cls.test_data_dir)
    
    @classmethod
    def _create_test_images(cls):
        """Create some dummy test images for testing"""
        # Create a few test images for each class
        for class_dir in cls.class_dirs:
            for i in range(5):
                # Create a simple pattern
                img = np.zeros((256, 256), dtype=np.uint8)
                if class_dir == "class1":
                    img[50:200, 50:200] = 255  # White square
                else:
                    img[50:200, 50:200] = 128  # Gray square
                img = Image.fromarray(img)
                img.save(os.path.join(cls.test_data_dir, class_dir, f"test_image_{i}.png"))
    
    def setUp(self):
        self.window = MainWindow()
    
    def tearDown(self):
        self.window.close()
    
    def test_window_creation(self):
        """Test that the window is created with correct title and tabs"""
        self.assertEqual(self.window.windowTitle(), "Fourier Optics AutoML for Photonics")
        self.assertEqual(self.window.tabs.count(), 2)
        self.assertEqual(self.window.tabs.tabText(0), "Data Analysis")
        self.assertEqual(self.window.tabs.tabText(1), "Inverse Design")
    
    def test_data_loading(self):
        """Test data loading functionality"""
        # Load data directly
        success = self.window.data_manager.load_data(self.test_data_dir)
        self.assertTrue(success)
        
        # Verify dataset info
        info = self.window.data_manager.get_dataset_info()
        self.assertEqual(info["num_samples"], 10)  # 5 images per class
        self.assertEqual(len(info["classes"]), 2)
    
    def test_preprocessing(self):
        """Test preprocessing functionality"""
        # First load data
        self.window.data_manager.load_data(self.test_data_dir)
        
        # Set target size
        target_size = (128, 128)
        
        # Run preprocessing
        success = self.window.data_manager.preprocess_data(target_size)
        self.assertTrue(success)
        
        # Verify processed data
        self.assertIsNotNone(self.window.data_manager.processed_data)
        self.assertEqual(self.window.data_manager.processed_data.shape[1:], target_size)
    
    def test_training_setup(self):
        """Test training setup"""
        # First load and preprocess data
        self.window.data_manager.load_data(self.test_data_dir)
        self.window.data_manager.preprocess_data()
        
        # Convert processed data to proper format for training
        processed_data = torch.from_numpy(self.window.data_manager.processed_data).float()
        targets = torch.tensor(self.window.data_manager.dataset.targets, dtype=torch.long)
        
        # Create a simple dataset
        dataset = TensorDataset(processed_data, targets)
        
        # Setup training
        train_loader, val_loader = self.window.training_manager.setup_training(
            dataset,
            model_type="FourierCNN",
            epochs=2
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
    
    def test_model_saving(self):
        """Test model saving functionality"""
        # First train a model
        self.window.data_manager.load_data(self.test_data_dir)
        self.window.data_manager.preprocess_data()
        
        # Convert processed data to proper format for training
        processed_data = torch.from_numpy(self.window.data_manager.processed_data).float()
        targets = torch.tensor(self.window.data_manager.dataset.targets, dtype=torch.long)
        dataset = TensorDataset(processed_data, targets)
        
        train_loader, val_loader = self.window.training_manager.setup_training(
            dataset,
            model_type="FourierCNN",
            epochs=2
        )
        
        # Train for a few epochs
        self.window.training_manager.train(train_loader, val_loader, 2)
        
        # Test saving
        test_save_path = os.path.join(self.test_data_dir, "test_model.pt")
        success = self.window.training_manager.save_model(test_save_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(test_save_path))
        
        # Clean up
        os.remove(test_save_path)

if __name__ == '__main__':
    unittest.main() 