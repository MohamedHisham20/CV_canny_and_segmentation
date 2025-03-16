import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog


def load_image(self):
    file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
    if file_name:
        self.image = cv2.imread(file_name)
        if self.image is None:
            return

        # Convert to RGB for display
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.modified_image = self.image.copy()
        self.show_image()


def show_image(self):
    if self.modified_image is None:
        return

    h, w, ch = self.modified_image.shape
    bytes_per_line = ch * w
    qimage = QImage(self.modified_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    # Store original pixmap for reset
    if self.original_pixmap is None:
        self.original_pixmap = pixmap

    # Scale pixmap to fit in label while maintaining aspect ratio
    scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                  Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation)

    self.image_label.setPixmap(scaled_pixmap)


def reset_image(self):
    if self.image is None:
        return

    self.modified_image = self.image.copy()
    self.show_image()

def convert_to_grey(self):
    if self.modified_image is None:
        return

    if self.modified_image.shape == 2:
        return
    grey_image = cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2GRAY)
    self.modified_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2RGB)
    self.show_image()

def save_image(self):
    if self.modified_image is None:
        return

    file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
    if file_name:
        cv2.imwrite(file_name, cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2BGR))