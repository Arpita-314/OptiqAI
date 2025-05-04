import sys
import os
import pytest
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from fourierlab.UI.gui.main_window import MainWindow
from fourierlab.core.pattern_generator import PatternGenerator
from fourierlab.core.phase_mask import PhaseMaskGenerator

# Create QApplication instance for testing
@pytest.fixture(scope="session")
def app():
    return QApplication(sys.argv)

@pytest.fixture
def main_window(app):
    window = MainWindow()
    window.show()
    return window

@pytest.fixture
def inverse_design_tab(main_window):
    # Switch to Inverse Design tab
    main_window.tabs.setCurrentIndex(1)
    return main_window.tabs.currentWidget()

class TestInverseDesignTab:
    def test_initial_state(self, inverse_design_tab):
        """Test initial state of the Inverse Design tab"""
        # Check target type combo box
        assert inverse_design_tab.target_type.currentText() == "Image"
        
        # Check pattern parameters
        assert inverse_design_tab.pattern_size.value() == 256
        assert inverse_design_tab.pattern_type.currentText() == "cross"
        assert inverse_design_tab.pattern_width.value() == 25
        assert inverse_design_tab.pattern_freq.value() == 10.0
        
        # Check generation parameters
        assert inverse_design_tab.wavelength.value() == 632.8
        assert inverse_design_tab.pixel_size.value() == 5.0
        assert inverse_design_tab.iterations.value() == 1000
        assert inverse_design_tab.learning_rate.value() == 0.01
        assert inverse_design_tab.optimizer_type.currentText() == "adam"
        assert inverse_design_tab.smoothness_weight.value() == 0.1
        assert inverse_design_tab.contrast_weight.value() == 0.05
        
        # Check button states
        assert inverse_design_tab.generate_btn.isEnabled()
        assert not inverse_design_tab.save_phase_btn.isEnabled()
        assert not inverse_design_tab.save_output_btn.isEnabled()
    
    def test_target_type_change(self, inverse_design_tab):
        """Test changing target type"""
        # Change to Pattern type
        inverse_design_tab.target_type.setCurrentText("Pattern")
        
        # Check control states
        assert not inverse_design_tab.upload_btn.isEnabled()
        assert inverse_design_tab.pattern_size.isEnabled()
        assert inverse_design_tab.pattern_type.isEnabled()
        assert inverse_design_tab.pattern_width.isEnabled()
        assert inverse_design_tab.pattern_freq.isEnabled()
        
        # Change back to Image type
        inverse_design_tab.target_type.setCurrentText("Image")
        
        # Check control states
        assert inverse_design_tab.upload_btn.isEnabled()
        assert not inverse_design_tab.pattern_size.isEnabled()
        assert not inverse_design_tab.pattern_type.isEnabled()
        assert not inverse_design_tab.pattern_width.isEnabled()
        assert not inverse_design_tab.pattern_freq.isEnabled()
    
    def test_pattern_generation(self, inverse_design_tab):
        """Test pattern generation and preview"""
        # Set to Pattern type
        inverse_design_tab.target_type.setCurrentText("Pattern")
        
        # Test different pattern types
        for pattern_type in ["cross", "circle", "square", "grating", "spiral"]:
            inverse_design_tab.pattern_type.setCurrentText(pattern_type)
            inverse_design_tab.update_pattern_preview()
            
            # Check that preview is updated
            assert inverse_design_tab.target_preview.pixmap() is not None
    
    def test_phase_mask_generation(self, inverse_design_tab, tmp_path):
        """Test phase mask generation process"""
        # Set to Pattern type and generate a simple pattern
        inverse_design_tab.target_type.setCurrentText("Pattern")
        inverse_design_tab.pattern_type.setCurrentText("cross")
        inverse_design_tab.update_pattern_preview()
        
        # Generate phase mask
        inverse_design_tab.generate_phase_mask()
        
        # Wait for generation to complete
        QTest.qWait(1000)  # Adjust wait time as needed
        
        # Check that results are displayed
        assert inverse_design_tab.phase_preview.pixmap() is not None
        assert inverse_design_tab.output_preview.pixmap() is not None
        
        # Check that save buttons are enabled
        assert inverse_design_tab.save_phase_btn.isEnabled()
        assert inverse_design_tab.save_output_btn.isEnabled()
    
    def test_save_functionality(self, inverse_design_tab, tmp_path):
        """Test saving phase mask and output"""
        # Generate a phase mask first
        inverse_design_tab.target_type.setCurrentText("Pattern")
        inverse_design_tab.pattern_type.setCurrentText("cross")
        inverse_design_tab.update_pattern_preview()
        inverse_design_tab.generate_phase_mask()
        
        # Wait for generation to complete
        QTest.qWait(1000)  # Adjust wait time as needed
        
        # Save phase mask
        phase_mask_path = tmp_path / "phase_mask.png"
        inverse_design_tab.save_phase_mask()
        assert phase_mask_path.exists()
        
        # Save output
        output_path = tmp_path / "output.png"
        inverse_design_tab.save_output()
        assert output_path.exists()
    
    def test_error_handling(self, inverse_design_tab):
        """Test error handling in various scenarios"""
        # Test generation without target
        inverse_design_tab.target_type.setCurrentText("Image")
        inverse_design_tab.generate_phase_mask()
        
        # Check that error message is shown
        # Note: This test might need adjustment based on how error messages are displayed
        
        # Test invalid parameters
        inverse_design_tab.wavelength.setValue(0)  # Invalid wavelength
        inverse_design_tab.generate_phase_mask()
        
        # Check that error message is shown
        # Note: This test might need adjustment based on how error messages are displayed

class TestPatternGenerator:
    def test_pattern_types(self):
        """Test generation of different pattern types"""
        generator = PatternGenerator()
        size = 256
        
        for pattern_type in ["cross", "circle", "square", "grating", "spiral"]:
            pattern = generator.generate_pattern(
                pattern_type=pattern_type,
                size=size,
                width=25,
                frequency=10
            )
            
            assert isinstance(pattern, np.ndarray)
            assert pattern.shape == (size, size)
            assert pattern.min() >= 0 and pattern.max() <= 1

class TestPhaseMaskGenerator:
    def test_optimization(self):
        """Test phase mask optimization process"""
        generator = PhaseMaskGenerator()
        
        # Create a simple target pattern
        target = np.zeros((256, 256))
        target[100:156, 100:156] = 1  # Simple square pattern
        
        # Set parameters
        params = {
            'wavelength': 632.8e-9,
            'pixel_size': 5e-6,
            'iterations': 100,
            'learning_rate': 0.01,
            'optimizer_type': 'adam',
            'smoothness_weight': 0.1,
            'contrast_weight': 0.05
        }
        generator.set_parameters(**params)
        
        # Track metrics
        metrics_history = []
        def callback(iteration, metrics):
            metrics_history.append(metrics)
        
        # Run optimization
        phase_mask, output = generator.optimize(target, callback=callback)
        
        # Check results
        assert isinstance(phase_mask, torch.Tensor)
        assert isinstance(output, torch.Tensor)
        assert phase_mask.shape == (256, 256)
        assert output.shape == (256, 256)
        assert len(metrics_history) > 0
        
        # Check that loss decreases
        initial_loss = metrics_history[0]['total_loss']
        final_loss = metrics_history[-1]['total_loss']
        assert final_loss <= initial_loss 