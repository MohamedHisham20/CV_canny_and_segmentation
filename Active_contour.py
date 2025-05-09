# main ui
import sys
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QRadioButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QPoint
from torch.nn.functional import threshold

from post_contouring import PostContour
from fpdf import FPDF
from edge_detector import canny_edge_detection
from hough_transform import detect_hough_lines, detect_hough_ellipse


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("CV_Task02.ui", self)
        # Connect buttons to their respective functions
        self.LoadImageButton.clicked.connect(self.load_image)
        self.SaveButton.clicked.connect(self.save_image)
        self.ResetButton.clicked.connect(self.reset)
        self.CircleContourButton.clicked.connect(self.draw_initial_contour)
        self.SquareContourButton.clicked.connect(self.draw_initial_contour)
        self.ApplyContourButton.clicked.connect(self.active_contour)
        self.CannyFilterButton.clicked.connect(self.canny_detection)
        self.SuperimposeButton.clicked.connect(self.hough_detection)

        self.LowThresholdHorizontalSlider.setValue(5)  # divide by 100
        self.LowThresholdHorizontalSlider.valueChanged.connect(self.update_low_threshold_label)
        self.HighThresholdHorizontalSlider.setValue(15)
        self.HighThresholdHorizontalSlider.valueChanged.connect(self.update_high_threshold_label)
        self.GaussianHorizontalSlider.setValue(5)
        self.GaussianHorizontalSlider.valueChanged.connect(self.update_gaussian_label)
        self.SigmaHorizontalSlider.setValue(14) # divide by 10
        self.SigmaHorizontalSlider.valueChanged.connect(self.update_sigma_label)
        self.SobelHorizontalSlider.setValue(3)
        self.SobelHorizontalSlider.valueChanged.connect(self.update_sobel_label)

        self.AlphaText.setText("0.1")
        self.BetaText.setText("0.1")
        self.GammaText.setText("1.0")
        self.IterationsText.setText("100")

        ### Chain_code related
        self.ChainCodeButton.clicked.connect(self.generate_contour_report)

        self.image = None
        self.output_image = None
        self.contour_position = QPoint(200, 200)
        self.contour_size = 50
        self.num_points = 20
        self.circle_points = []
        self.square_points = []
        self.dragging = False
        self.drag_offset = QPoint(0, 0)
        self.create_contour_points()

        self.canny_image = None

        self.chain_points = []

    def update_low_threshold_label(self):
        self.Lowthreshold.setText(str(self.LowThresholdHorizontalSlider.value()))

    def update_high_threshold_label(self):
        self.Highthreshold.setText(str(self.HighThresholdHorizontalSlider.value()))

    def update_gaussian_label(self):
        self.Gaussian.setText(str(self.GaussianHorizontalSlider.value()))

    def update_sigma_label(self):
        self.Sigma.setText(str(self.SigmaHorizontalSlider.value() / 10))

    def update_sobel_label(self):
        self.Sobel.setText(str(self.SobelHorizontalSlider.value()))

    def create_contour_points(self):
        """Generate dynamic points for the circle and square."""
        self.num_points = max(20, int(self.contour_size * 2))
        self.circle_points = []
        self.square_points = []

        # Generate circle points
        for i in range(self.num_points):
            angle = (2 * np.pi * i) / self.num_points
            x = self.contour_position.x() + self.contour_size * np.cos(angle)
            y = self.contour_position.y() + self.contour_size * np.sin(angle)
            self.circle_points.append((int(x), int(y)))

        # Generate square points
        step = self.contour_size * 2 / (self.num_points // 4)
        for i in range(self.num_points // 4):
            x = self.contour_position.x() - self.contour_size + i * step
            y = self.contour_position.y() - self.contour_size
            self.square_points.append((int(x), int(y)))
        for i in range(self.num_points // 4):
            x = self.contour_position.x() + self.contour_size
            y = self.contour_position.y() - self.contour_size + i * step
            self.square_points.append((int(x), int(y)))
        for i in range(self.num_points // 4):
            x = self.contour_position.x() + self.contour_size - i * step
            y = self.contour_position.y() + self.contour_size
            self.square_points.append((int(x), int(y)))
        for i in range(self.num_points // 4):
            x = self.contour_position.x() - self.contour_size
            y = self.contour_position.y() + self.contour_size - i * step
            self.square_points.append((int(x), int(y)))

    def canny_detection(self):
        if self.image is None:
            print("Error: No image loaded.")
            return
        #method 1
        self.canny_image = canny_edge_detection(self.image,
                                                gaussian_kernel_size=self.GaussianHorizontalSlider.value(),
                                                sigma=self.SigmaHorizontalSlider.value() / 10,
                                                low_threshold=self.LowThresholdHorizontalSlider.value(),
                                                high_threshold=self.HighThresholdHorizontalSlider.value() ,
                                                sobel_kernel_size=self.SobelHorizontalSlider.value())

        # method 2
        # self.canny_image = canny_edge_detection(image=self.image)

        # method 3
        # self.canny_image = canny_edge_detection(self.image,
        #                                         gaussian_kernel_size=self.GaussianHorizontalSlider.value(),
        #                                         sigma=self.SigmaHorizontalSlider.value() / 10,
        #                                         low_threshold=self.LowThresholdHorizontalSlider.value() / 100,
        #                                         high_threshold=self.HighThresholdHorizontalSlider.value() / 100,
        #                                         sobel_kernel_size=self.SobelHorizontalSlider.value())

        self.display_image(self.canny_image, self.OutputImage)
        self.output_image = self.canny_image

    def hough_detection(self):
        print("Checking if canny_image is None...")
        if self.canny_image is None:
            print("Error: No image loaded.")
            return

        print("Checking which filter is selected...")
        if self.EllipesFilterButton.isChecked():
            print("Ellipse filter selected.")
            hough_image = detect_hough_ellipse(self.canny_image, shape='ellipse')
        elif self.CircleFilterButton.isChecked():
            print("Circle filter selected.")
            hough_image = detect_hough_ellipse(self.canny_image, shape='circle')
        elif self.LineFilterButton.isChecked():
            print("Line filter selected.")
            hough_image, lines = detect_hough_lines(self.canny_image)
        else:
            print("Error: Please select a filter type.")
            return

        print("Displaying image...")
        self.display_image(hough_image, self.OutputImage)
        self.output_image = hough_image
        print("Processing completed successfully.")

    def draw_initial_contour(self):
        if self.image is None:
            return

        display_image = self.image.copy()
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        overlay = display_image.copy()

        self.create_contour_points()

        if self.CircleContourButton.isChecked():
            for i in range(len(self.circle_points)):
                cv2.line(overlay, self.circle_points[i], self.circle_points[(i+1) % len(self.circle_points)], (0, 255, 0), 2)
        elif self.SquareContourButton.isChecked():
            for i in range(len(self.square_points)):
                cv2.line(overlay, self.square_points[i], self.square_points[(i+1) % len(self.square_points)], (0, 255, 0), 2)

        cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0, display_image)
        self.display_image(display_image, self.InputImage)

    def display_image(self, image, widget):
        if image is None:
            return

        # Convert image for display
        if len(image.shape) == 2:
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        
        # Set up the widget with the image
        if widget == self.InputImage:
            if not hasattr(self, 'input_image_label'):
                self.input_image_label = QLabel(widget)
                self.input_image_label.setScaledContents(True)
                self.input_image_label.setGeometry(0, 0, widget.width(), widget.height())
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.input_image_label)
                widget.setLayout(layout)
            self.input_image_label.setPixmap(pixmap)
        elif widget == self.OutputImage:
            if not hasattr(self, 'output_image_label'):
                self.output_image_label = QLabel(widget)
                self.output_image_label.setScaledContents(True)
                self.output_image_label.setGeometry(0, 0, widget.width(), widget.height())
                layout = QVBoxLayout(widget)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.output_image_label)
                widget.setLayout(layout)
            self.output_image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_offset = event.pos() - self.contour_position

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.contour_position = event.pos() - self.drag_offset
            self.draw_initial_contour()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.contour_size += 5
        else:
            self.contour_size -= 5
        self.contour_size = max(10, min(self.contour_size, 200))
        self.draw_initial_contour()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                print("Error: Failed to load image.")
                return
            self.display_image(self.image, self.InputImage)

    def save_image(self):
        if self.output_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.output_image)
                print(f"Image saved to {file_path}")
        else:
            print("Error: No output image to save.")


    def reset(self):
        self.image = None
        self.output_image = None
        self.circle_points.clear()
        self.square_points.clear()
        print("Application reset successfully.")



    def compute_gradient_edges(self, image_array, sigma=1, threshold=20):
        """Compute gradient magnitude and detect edge points in the image.
        Set edge points' external energy to -1000 before returning.
        """
        # Ensure the image is grayscale
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        image_array = image_array.astype(np.float32)
        blurred = gaussian_filter(image_array, sigma=sigma)
        gx = np.gradient(blurred, axis=1)
        gy = np.gradient(blurred, axis=0)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Thresholding to extract edges
        edges = gradient_magnitude > threshold
        edge_points = np.column_stack(np.where(edges))
        external_energy = -gradient_magnitude  # Negative attracts contour to edges
        min_energy = np.min(external_energy)
        max_energy = np.max(external_energy)
        if max_energy - min_energy > 1e-6:
            external_energy = (external_energy - min_energy) / (max_energy - min_energy)
        
        # Set external energy at edge points to -1000
        external_energy[edge_points[:, 0], edge_points[:, 1]] = -1000
        
        return edge_points, external_energy


    def redistribute_points(self, contour, num_points):
        """Redistribute contour points to maintain equal spacing."""
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.vstack((x_new, y_new)).T


    def active_contour(self):
        if self.image is None:
            print("Error: No image loaded.")
            return
        
        # Get parameters from text fields
        try:
            self.alpha = float(self.AlphaText.text())
            self.beta = float(self.BetaText.text())
            self.gamma = float(self.GammaText.text())
            self.max_iterations = int(self.IterationsText.text())
        except ValueError:
            print("Error: Invalid parameter values. Please enter numbers.")
            return
        
        # Get initial contour based on selected shape
        if self.CircleContourButton.isChecked():
            initial_contour = np.array(self.circle_points)
        elif self.SquareContourButton.isChecked():
            initial_contour = np.array(self.square_points)
        else:
            print("Error: Please select a contour shape (Circle or Square).")
            return
        
        # Calculate edge points and external energy
        edge_points, external_energy = self.compute_gradient_edges(self.image)
        
        # Perform active contour algorithm
        num_points = initial_contour.shape[0]
        A = np.zeros((num_points, num_points))
        
        for i in range(num_points):
            A[i, i] = 2 * self.alpha + 6 * self.beta
            A[i, (i + 1) % num_points] = -self.alpha - 4 * self.beta
            A[i, (i - 1) % num_points] = -self.alpha - 4 * self.beta
            A[i, (i + 2) % num_points] = self.beta
            A[i, (i - 2) % num_points] = self.beta
        
        A_inv = np.linalg.inv(A + self.gamma * np.eye(num_points))
        contour = initial_contour.copy()

        for iteration in range(self.max_iterations):
            fx = np.gradient(external_energy, axis=1)
            fy = np.gradient(external_energy, axis=0)

            fx = fx / np.max(np.abs(fx))
            fy = fy / np.max(np.abs(fy))

            x_coords = contour[:, 0].astype(int)
            y_coords = contour[:, 1].astype(int)
            x_coords = np.clip(x_coords, 0, self.image.shape[1] - 1)
            y_coords = np.clip(y_coords, 0, self.image.shape[0] - 1)

            external_forces_x = fx[y_coords, x_coords]
            external_forces_y = fy[y_coords, x_coords]
            external_forces = np.vstack((external_forces_x, external_forces_y)).T
            
            contour = np.dot(A_inv, self.gamma * contour - external_forces)
            
            if iteration % 10 == 0:
                contour = self.redistribute_points(contour, num_points)
                # print(f"Iteration {iteration}: Contour Points:\n", contour)
            
            # Check if any point reaches the edge (-1000 energy)
            # if np.any(external_energy[y_coords, x_coords] < -999):
            #     print(f"Contour reached edge (energy < -999) at iteration {iteration + 1}. Stopping.")
            #     break
        
        # Draw the final contour on the output image
        result_image = self.image.copy()
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        
        # Draw the final contour in red
        contour_points = contour.astype(np.int32)
        for i in range(len(contour_points)):
            cv2.line(result_image, 
                    tuple(contour_points[i]), 
                    tuple(contour_points[(i+1) % len(contour_points)]), 
                    (0, 0, 255), 2)
        
        self.chain_points = contour_points
        # Store the output image
        self.output_image = result_image
        
        # Display the output image
        self.display_image(self.output_image, self.OutputImage)
        
        print("Active contour completed successfully.")

    def generate_contour_report(self):
        
        chain_code = PostContour.generate_chain_code(self.chain_points)
        area = round(PostContour.calculate_area(self.chain_points))
        percent = round((area / (self.image.shape[0] * self.image.shape[1])) * 100, 2)
        perimeter = round(PostContour.calculate_perimeter(self.chain_points))
        screenshot = self.output_image  
        
        image_filename = "contour_image.png"
        if isinstance(screenshot, (list, tuple, cv2.UMat)) or hasattr(screenshot, "shape"):
            cv2.imwrite(image_filename, screenshot)
        else:
            print("Error: Invalid image data.")
            return
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, "Contour Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Contour Image:", ln=True)
        pdf.image(image_filename, x=10, y=pdf.get_y(), w=100)
        pdf.ln(80)  

        pdf.cell(200, 10, f"Area: {area} Pixels, equal to {percent}% of total image size.", ln=True)
        pdf.cell(200, 10, f"Perimeter: {perimeter} Pixels.", ln=True)
        

        formatted_chain_code = " ".join(str(x) for x in chain_code)
        pdf.multi_cell(0, 10, f"Chain Code: [{formatted_chain_code}]")
        output_pdf = "contour_report.pdf"
        pdf.output(output_pdf)
        print(f"Report saved as {output_pdf}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

#