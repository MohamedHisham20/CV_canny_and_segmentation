import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QApplication
from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
import general_functions as gf


class CannyEdgeDetection(QWidget):
    def __init__(self):
        super().__init__()

        # Image data
        self.image = None
        self.modified_image = None
        self.original_pixmap = None

        # Setup UI
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        # Buttons layout
        button_layout = QHBoxLayout()

        # Load image button
        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)
        button_layout.addWidget(load_button)

        # Reset button
        reset_button = QPushButton('Reset Image')
        reset_button.clicked.connect(self.reset_image)
        button_layout.addWidget(reset_button)

        # Convert to grey button
        grey_button = QPushButton('Convert to Grey')
        grey_button.clicked.connect(self.convert_to_grey)
        button_layout.addWidget(grey_button)

        # Canny edge detection button
        canny_button = QPushButton('Detect Edges & Shapes')
        canny_button.clicked.connect(self.canny_edge_detection)
        button_layout.addWidget(canny_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            self.image = cv2.imread(filename)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.modified_image = self.image.copy()
            self.display_image(self.image)
            self.original_pixmap = self.image_label.pixmap()

    def display_image(self, image):
        if image is not None:
            height, width = image.shape[:2]
            if len(image.shape) == 3:  # Color image
                bytes_per_line = 3 * width
                q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:  # Grayscale image
                q_img = QImage(image.data, width, height, width, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio))

    def show_image(self):
        self.display_image(self.modified_image)

    def reset_image(self):
        if self.image is not None:
            self.modified_image = self.image.copy()
            self.display_image(self.image)

    def convert_to_grey(self):
        if self.modified_image is not None:
            if len(self.modified_image.shape) == 3:
                self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2GRAY)
                self.display_image(self.modified_image)

    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur to an image using custom implementation."""
        # Create a 1D Gaussian kernel
        k = kernel_size // 2
        x = np.linspace(-k, k, kernel_size)
        kernel_1d = np.exp(-0.5 * np.square(x) / np.square(sigma))
        kernel_1d = kernel_1d / np.sum(kernel_1d)

        # Create 2D kernel
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Apply convolution
        padded = np.pad(image, ((k, k), (k, k)), mode='reflect')
        output = np.zeros_like(image, dtype=np.float64)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(padded[i:i + kernel_size, j:j + kernel_size] * kernel_2d)

        return output

    def sobel_filters(self, image):
        """Apply Sobel filters to compute gradients."""
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Pad the image
        padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
        gradient_x = np.zeros_like(image, dtype=np.float64)
        gradient_y = np.zeros_like(image, dtype=np.float64)

        # Apply convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                gradient_x[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel_x)
                gradient_y[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kernel_y)

        return gradient_x, gradient_y

    def non_max_suppression(self, magnitude, direction):
        """Apply non-maximum suppression to thin edges."""
        M, N = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Quantize direction angles to 0, 45, 90, 135 degrees
        angle = np.rad2deg(direction) % 180
        angle_quantized = np.zeros_like(angle, dtype=np.uint8)

        # 0 degrees (horizontal)
        angle_quantized[(angle >= 0) & (angle < 22.5) | (angle >= 157.5) & (angle < 180)] = 0
        # 45 degrees (diagonal)
        angle_quantized[(angle >= 22.5) & (angle < 67.5)] = 45
        # 90 degrees (vertical)
        angle_quantized[(angle >= 67.5) & (angle < 112.5)] = 90
        # 135 degrees (diagonal)
        angle_quantized[(angle >= 112.5) & (angle < 157.5)] = 135

        # Pad the magnitude array to handle edge pixels
        padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if angle_quantized[i - 1, j - 1] == 0:  # horizontal
                    prev = padded_magnitude[i, j - 1]
                    next = padded_magnitude[i, j + 1]
                elif angle_quantized[i - 1, j - 1] == 45:  # diagonal (bottom-left to top-right)
                    prev = padded_magnitude[i + 1, j - 1]
                    next = padded_magnitude[i - 1, j + 1]
                elif angle_quantized[i - 1, j - 1] == 90:  # vertical
                    prev = padded_magnitude[i - 1, j]
                    next = padded_magnitude[i + 1, j]
                else:  # 135 degrees (top-left to bottom-right)
                    prev = padded_magnitude[i - 1, j - 1]
                    next = padded_magnitude[i + 1, j + 1]

                # If current pixel is a local maximum, keep it
                if padded_magnitude[i, j] >= prev and padded_magnitude[i, j] >= next:
                    suppressed[i - 1, j - 1] = magnitude[i - 1, j - 1]

        return suppressed

    def hysteresis_thresholding(self, image, low_ratio=0.05, high_ratio=0.15):
        """Apply hysteresis thresholding to detect edges."""
        high_threshold = image.max() * high_ratio
        low_threshold = high_threshold * low_ratio

        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)

        # Create a mask
        edges = np.zeros_like(image, dtype=bool)
        edges[strong_edges] = True

        # Find weak edges connected to strong edges
        rows, cols = np.where(weak_edges)
        for r, c in zip(rows, cols):
            if self.is_connected_to_strong(r, c, edges, weak_edges):
                edges[r, c] = True

        return edges

    def is_connected_to_strong(self, row, col, strong_edges, weak_edges, max_dist=2):
        """Check if a weak edge pixel is connected to any strong edge pixel."""
        h, w = strong_edges.shape

        # Check in neighborhood (up to max_dist pixels away)
        for i in range(max(0, row - max_dist), min(h, row + max_dist + 1)):
            for j in range(max(0, col - max_dist), min(w, col + max_dist + 1)):
                if strong_edges[i, j]:
                    # There's a strong edge nearby
                    return True

        return False

    def hough_transform_lines(self, edges, threshold=50):
        """Detect lines using Hough transform."""
        h, w = edges.shape
        diag_len = int(np.ceil(np.sqrt(h * h + w * w)))

        # Hough space: theta (-90 to 90 degrees), rho (-diag_len to diag_len)
        thetas = np.deg2rad(np.arange(-90, 90))
        rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)

        # Initialize accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(edges)

        # Vote in the accumulator
        for y, x in zip(y_idxs, x_idxs):
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                accumulator[rho, theta_idx] += 1

        # Find peaks in the accumulator
        lines = []
        for rho_idx, theta_idx in zip(*np.where(accumulator > threshold)):
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            lines.append((rho, theta))

        return lines

    def hough_transform_circles(self, edges, min_radius=10, max_radius=100, threshold=30):
        """Detect circles using Hough transform."""
        h, w = edges.shape
        y_idxs, x_idxs = np.nonzero(edges)

        # Possible radii
        radii = np.arange(min_radius, max_radius + 1)

        # Initialize accumulator for each radius
        circles = []
        for radius in radii:
            accumulator = np.zeros((h, w), dtype=np.uint64)

            # For each edge point
            for y, x in zip(y_idxs, x_idxs):
                # For each possible angle
                for angle in range(0, 360, 5):  # Step by 5 degrees for efficiency
                    a = int(x - radius * np.cos(np.deg2rad(angle)))
                    b = int(y - radius * np.sin(np.deg2rad(angle)))

                    if 0 <= a < w and 0 <= b < h:
                        accumulator[b, a] += 1

            # Find peaks for this radius
            for cy, cx in zip(*np.where(accumulator > threshold)):
                circles.append((cx, cy, radius))

        # Filter overlapping circles
        filtered_circles = []
        for circle in sorted(circles, key=lambda x: x[2], reverse=True):  # Sort by radius (largest first)
            cx, cy, r = circle
            if not any(np.sqrt((cx - x) ** 2 + (cy - y) ** 2) < (r + r2) / 2 for x, y, r2 in filtered_circles):
                filtered_circles.append(circle)
                if len(filtered_circles) >= 10:  # Limit to top 10 circles
                    break

        return filtered_circles

    def fit_ellipses(self, edges, threshold=100):
        """Fit ellipses to edge points using direct least squares method."""
        # Find connected components in the edge image
        num_labels, labels = cv2.connectedComponents(edges.astype(np.uint8))

        ellipses = []
        for label in range(1, num_labels):  # Skip background (0)
            points = np.array(np.where(labels == label)).T

            # Need at least 5 points to fit an ellipse
            if len(points) < 5:
                continue

            # Swap x and y coordinates for OpenCV fitting
            points = points[:, ::-1]

            # Use OpenCV's fitEllipse as a fallback since ellipse fitting is complex
            if len(points) >= 5:
                try:
                    ellipse = cv2.fitEllipse(points.astype(np.float32))
                    # Filter by size
                    center, axes, angle = ellipse
                    if min(axes) > 5 and max(axes) < 150:  # Reasonable ellipse size
                        ellipses.append(ellipse)
                except:
                    pass

        # Limit number of ellipses
        if len(ellipses) > 10:
            # Sort by area (largest first)
            ellipses = sorted(ellipses, key=lambda e: e[1][0] * e[1][1], reverse=True)[:10]

        return ellipses

    def canny_edge_detection(self):
        """Apply Canny edge detection algorithm and detect shapes."""
        if self.modified_image is None:
            return

        # Convert to grayscale if needed
        if len(self.modified_image.shape) == 3:
            grey_image = cv2.cvtColor(self.modified_image, cv2.COLOR_RGB2GRAY)
        else:
            grey_image = self.modified_image

        # Step 1: Apply Gaussian Blur to reduce noise
        blurred_image = self.gaussian_blur(grey_image.astype(np.float64), 5, 1.4)

        # Step 2: Compute gradients using Sobel filters
        gradient_x, gradient_y = self.sobel_filters(blurred_image)

        # Step 3: Calculate gradient magnitude and direction
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        # Normalize magnitude to 0-255 range
        gradient_magnitude = gradient_magnitude * 255.0 / (gradient_magnitude.max() or 1)

        # Step 4: Apply non-maximum suppression
        suppressed = self.non_max_suppression(gradient_magnitude, gradient_direction)

        # Step 5: Apply hysteresis thresholding
        edges = self.hysteresis_thresholding(suppressed)
        edge_image = edges.astype(np.uint8) * 255

        # Create RGB image for visualization
        result_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2RGB)

        # Step 6: Detect lines
        lines = self.hough_transform_lines(edges, threshold=40)

        # Step 7: Detect circles
        circles = self.hough_transform_circles(edges, min_radius=15, max_radius=100, threshold=25)

        # Step 8: Detect ellipses
        ellipses = self.fit_ellipses(edge_image)

        # Step 9: Draw original image
        result_image = self.image.copy() if len(self.image.shape) == 3 else cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        # Step 10: Draw detected shapes on the image
        # Draw lines (in red)
        h, w = edges.shape
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw circles (in green)
        for x, y, r in circles:
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)

        # Draw ellipses (in blue)
        for ellipse in ellipses:
            cv2.ellipse(result_image, ellipse, (0, 0, 255), 2)

        # Draw edges (in white, semi-transparent)
        edge_overlay = result_image.copy()
        edge_rgb = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)
        mask = edge_image > 0
        edge_overlay[mask] = (255, 255, 255)
        result_image = cv2.addWeighted(result_image, 0.7, edge_overlay, 0.3, 0)

        # Update the modified image
        self.modified_image = result_image
        self.show_image()

        return result_image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Canny Edge Detection & Shape Detection')
        self.setGeometry(100, 100, 800, 600)

        # Create the Canny widget
        self.canny_widget = CannyEdgeDetection()
        self.setCentralWidget(self.canny_widget)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())