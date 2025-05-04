from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QVBoxLayout,
    QLabel, QPushButton
)

class FourierLabGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FourierLab - No-Code Optics Analysis")
        self.setup_ui()

    def setup_ui(self):
        self.upload_btn = QPushButton("Upload Images", self)
        self.upload_btn.clicked.connect(self.handle_upload)
        
        self.status_label = QLabel("Ready for upload", self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.status_label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def handle_upload(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Optical Images", 
            "", "Images (*.png *.jpg *.tif)"
        )
        if files:
            self.status_label.setText(f"Processing {len(files)} images...")
            self.start_analysis(files)