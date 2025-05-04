from fourierlab.UI.gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys
import cupy as cp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    cp.show_config()
    sys.exit(app.exec_()) 