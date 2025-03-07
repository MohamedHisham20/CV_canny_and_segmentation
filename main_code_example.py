from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QApplication, QMainWindow
import sys

from general_functions import load_image, show_image, reset_image


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.modified_image = None
        self.original_pixmap = None
        self.image_label = QLabel()

        # Add buttons for load, show, and reset
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)

        # Set up layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.load_image_button)
        self.layout.addWidget(self.reset_button)
        self.setLayout(self.layout)

    def load_image(self):
        load_image(self)

    def show_image(self):
        show_image(self)

    def reset_image(self):
        reset_image(self)


# Test run
app = QApplication([])
window = QMainWindow()
processor = ImageProcessor()
window.setCentralWidget(processor)
window.show()
sys.exit(app.exec())