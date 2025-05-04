from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QStatusBar
from PyQt5.QtCore import Qt
import torch

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FourierLab")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add CUDA status label
        self.cuda_status = QLabel()
        self.update_cuda_status()
        layout.addWidget(self.cuda_status)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def update_cuda_status(self):
        """Update CUDA status information"""
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            self.cuda_status.setText(f"CUDA Status: Active\nDevice: {device_name}\nCUDA Version: {cuda_version}")
            self.cuda_status.setStyleSheet("color: green; padding: 10px;")
        else:
            self.cuda_status.setText("CUDA Status: Not Available\nRunning in CPU mode")
            self.cuda_status.setStyleSheet("color: red; padding: 10px;") 