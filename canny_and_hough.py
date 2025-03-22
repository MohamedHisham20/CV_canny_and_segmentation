import cv2
import numpy as np
from matplotlib import pyplot as plt

import general_functions as gf

def show_image(widget):
    gf.show_image(widget)

# def detect_hough_ellipse(image, parameters=None):
#     if image is None:
#         print("Error: No image loaded for Hough transform.")
#         return
#
#     params = {
#         'minMajorAxis': 5,
#         'maxMajorAxis': 100,
#         'rotation': -360,
#         'rotationSpan': 360,
#         'minAspectRatio': 0.3,
#         'randomize': 5,
#         'numBest': 10,
#         'uniformWeights': True,
#         'smoothStddev': 1,
#         'max_points': 8000
#     }
#     if parameters is not None:
#         params.update(parameters)
#
#     test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
#         image.shape) == 3 else image
#     ellipses = ellipse_detection(test_image, params, verbose=True)
#     result_image = draw_ellipses(test_image, ellipses)
#     image = result_image
#     return image


def canny_edge_detection(image, gaussian_kernel_size=5, sigma=1.4, low_threshold=0.05, high_threshold=0.15, sobel_kernel_size=3):
    if image is None:
        print("Error: No image loaded for Canny.")
        return

    original_image = image.copy()
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(
        image.shape) == 3 else image

    blurred_image = gaussian_blur(grey_image.astype(np.float64), gaussian_kernel_size, sigma)
    gradient_x, gradient_y,image = sobel_filters(blurred_image, threshold=int(high_threshold), kernel_size=sobel_kernel_size)
    # gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    # gradient_direction = np.arctan2(gradient_y, gradient_x)
    # gradient_magnitude = gradient_magnitude * 255.0 / (gradient_magnitude.max() or 1)

    # suppressed_edges = non_max_suppression(gradient_magnitude, gradient_direction)
    # final_edges = hysteresis_thresholding(suppressed_edges, low_ratio=low_threshold, high_ratio=high_threshold)
    #
    # image = (final_edges * 255).astype(np.uint8)
    return image


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    k = kernel_size // 2
    x = np.linspace(-k, k, kernel_size)
    kernel_1d = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel_1d /= np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    padded = np.pad(image, ((k, k), (k, k)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i + kernel_size, j:j + kernel_size] * kernel_2d)

    return output


def sobel_filters(image, threshold=20, kernel_size=3):
    """Apply Sobel edge detection from scratch with variable kernel size"""
    # Get image dimensions
    rows, cols = image.shape

    # Create Sobel kernels based on kernel size
    if kernel_size == 3:
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    else:
        # Generate larger Sobel-like kernels
        # This is a simple approximation - more sophisticated kernels could be used
        mid = kernel_size // 2
        Kx = np.zeros((kernel_size, kernel_size))
        Ky = np.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                x_dist = j - mid
                y_dist = i - mid
                if x_dist != 0:
                    Kx[i, j] = x_dist / abs(x_dist) * (mid - abs(y_dist)) if abs(y_dist) <= mid else 0
                if y_dist != 0:
                    Ky[i, j] = y_dist / abs(y_dist) * (mid - abs(x_dist)) if abs(x_dist) <= mid else 0

    # Initialize gradient images
    Ix = np.zeros_like(image, dtype=np.float64)
    Iy = np.zeros_like(image, dtype=np.float64)

    # Apply convolution
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant')

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            Ix[i, j] = np.sum(region * Kx)
            Iy[i, j] = np.sum(region * Ky)

    # Calculate gradient magnitude
    G = np.sqrt(Ix ** 2 + Iy ** 2)
    G = np.clip(G, 0, 255).astype(np.uint8)

    # Apply threshold and convert to binary
    for i in range(rows):
        for j in range(cols):
            if G[i, j] < threshold:
                G[i, j] = 0
            else:
                G[i, j] = 255


    return Ix,Iy,G


def non_max_suppression(magnitude, direction):
    M, N = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = np.rad2deg(direction) % 180
    angle_quantized = np.zeros_like(angle, dtype=np.uint8)

    angle_quantized[(angle >= 0) & (angle < 22.5) | (angle >= 157.5) & (angle < 180)] = 0
    angle_quantized[(angle >= 22.5) & (angle < 67.5)] = 45
    angle_quantized[(angle >= 67.5) & (angle < 112.5)] = 90
    angle_quantized[(angle >= 112.5) & (angle < 157.5)] = 135

    padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            prev, next = 0, 0
            if angle_quantized[i - 1, j - 1] == 0:
                prev = padded_magnitude[i, j - 1]
                next = padded_magnitude[i, j + 1]
            elif angle_quantized[i - 1, j - 1] == 45:
                prev = padded_magnitude[i + 1, j - 1]
                next = padded_magnitude[i - 1, j + 1]
            elif angle_quantized[i - 1, j - 1] == 90:
                prev = padded_magnitude[i - 1, j]
                next = padded_magnitude[i + 1, j]
            else:
                prev = padded_magnitude[i - 1, j - 1]
                next = padded_magnitude[i + 1, j + 1]

            if padded_magnitude[i, j] >= prev and padded_magnitude[i, j] >= next:
                suppressed[i - 1, j - 1] = magnitude[i - 1, j - 1]

    return suppressed


def hysteresis_thresholding(image, low_ratio=0.05, high_ratio=0.15):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    edges = np.zeros_like(image, dtype=bool)
    edges[strong_edges] = True

    rows, cols = np.where(weak_edges)
    for r, c in zip(rows, cols):
        if is_connected_to_strong(r, c, edges, weak_edges):
            edges[r, c] = True

    return edges


def is_connected_to_strong(row, col, strong_edges, weak_edges, max_dist=2):
    h, w = strong_edges.shape
    for i in range(max(0, row - max_dist), min(h, row + max_dist + 1)):
        for j in range(max(0, col - max_dist), min(w, col + max_dist + 1)):
            if strong_edges[i, j]:
                return True
    return False
