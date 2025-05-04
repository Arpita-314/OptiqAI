import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import pyqtgraph as pg

def main():
    """Main entry point for the quantum optics GUI"""
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Import here to avoid circular import
    from .main_window import MainWindow
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_()) 