import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import cos, sin, pi


def line_hough_transform(img, theta_res=1, rho_res=1):
    # Get image dimensions
    height, width = img.shape

    # Calculate diagonal length of the image
    diag_len = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    # Create parameter spaces: rho range is twice the diagonal length to accommodate negative values
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    num_thetas = len(thetas)
    num_rhos = 2 * diag_len // rho_res
    rhos = np.linspace(-diag_len, diag_len, num_rhos)

    # Create the accumulator array - using int32 instead of uint64
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

    # Find edge points
    y_idxs, x_idxs = np.nonzero(img)

    # For each edge point, calculate rho for every theta and increment accumulator
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Process points in batches to manage memory usage
    batch_size = 1000
    for i in range(0, len(y_idxs), batch_size):
        y_batch = y_idxs[i:i + batch_size]
        x_batch = x_idxs[i:i + batch_size]

        for j, (cos_t, sin_t) in enumerate(zip(cos_thetas, sin_thetas)):
            # Calculate rho values for this theta across all points in batch
            rho_vals = x_batch * cos_t + y_batch * sin_t

            # Map rho values to accumulator indices
            rho_indices = np.floor((rho_vals + diag_len) / (2 * diag_len / num_rhos)).astype(int)

            # Constrain indices to valid range
            valid_indices = (rho_indices >= 0) & (rho_indices < num_rhos)
            valid_rhos = rho_indices[valid_indices]

            # Increment accumulator
            if len(valid_rhos) > 0:
                unique_rhos, counts = np.unique(valid_rhos, return_counts=True)
                # Use np.add.at for safe accumulation
                np.add.at(accumulator, (unique_rhos, np.full_like(unique_rhos, j)), counts)

    return accumulator, thetas, rhos


def detect_lines_from_accumulator(accumulator, thetas, rhos, threshold, nms_window_size=None):
    """
    Detect lines from the accumulator with optional non-maximum suppression
    """
    if nms_window_size:
        # Apply non-maximum suppression to the accumulator
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(accumulator, size=nms_window_size)
        peaks = (accumulator == local_max) & (accumulator >= threshold)
    else:
        # Simple thresholding
        peaks = accumulator >= threshold

    # Get the indices of the peaks
    rho_idxs, theta_idxs = np.nonzero(peaks)

    # Create a list of detected lines (rho, theta)
    lines = []
    for rho_idx, theta_idx in zip(rho_idxs, theta_idxs):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta, accumulator[rho_idx, theta_idx]))  # Include vote count

    # Sort lines by vote count in descending order
    lines.sort(key=lambda x: x[2], reverse=True)

    return lines


def draw_lines(img, lines, max_lines=None):
    """Draw detected lines on an image"""
    # Create a color copy of the image
    img_out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    # Limit the number of lines to draw if specified
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]

    # Draw each line
    for rho, theta, votes in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Calculate endpoints of the line segment
        x1 = int(x0 + 1500 * (-b))  # Using larger values for longer lines
        y1 = int(y0 + 1500 * (a))
        x2 = int(x0 - 1500 * (-b))
        y2 = int(y0 - 1500 * (a))

        # Draw the line
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img_out


def visualize_accumulator(accumulator, title="Hough Space"):
    """Visualize the Hough accumulator space"""
    plt.figure(figsize=(10, 8))
    plt.imshow(accumulator, cmap='hot', aspect='auto')
    plt.title(title)
    plt.xlabel('Theta (index)')
    plt.ylabel('Rho (index)')
    plt.colorbar(label='Votes')
    plt.tight_layout()
    plt.show()


# Main processing function
def detect_hough_lines(image, params=None):
    # # Read the image
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if params is None:
        params = {}

    # Extract parameters with defaults
    theta_res = params.get('theta_res', 0.5)
    rho_res = params.get('rho_res', 1)
    threshold = params.get('threshold', 50)
    max_lines = params.get('max_lines', 15)
    enhance_edges = params.get('enhance_edges', True)
    show_accumulator = params.get('show_accumulator', False)

    if image is None:
        raise ValueError("Invalid image provided")

    # enhance edges
    edges = image
    if enhance_edges:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(image, kernel, iterations=1)

    # Apply Hough transform
    accumulator, thetas, rhos = line_hough_transform(edges, theta_res, rho_res)

    # Show accumulator if requested
    if show_accumulator:
        visualize_accumulator(accumulator)

    # Detect lines
    lines = detect_lines_from_accumulator(accumulator, thetas, rhos, threshold, nms_window_size=(15, 15))

    # Draw lines on the original image
    result = draw_lines(image, lines, max_lines)

    # # Display results
    # plt.figure(figsize=(15, 10))
    #
    # plt.subplot(131)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.subplot(132)
    # plt.imshow(edges, cmap='gray')
    # plt.title('Edge Detection')
    # plt.axis('off')
    #
    # plt.subplot(133)
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title(f'Detected Lines (top {max_lines if max_lines else "all"})')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return result, lines


# Example usage
# if __name__ == "__main__":
#     image_path = "images/vert_horz_lines_canna.png"
#     result, lines = process_image(
#         image_path,
#         theta_res=0.5,  # Higher angular resolution
#         rho_res=1,
#         threshold=50,  # Lower threshold to detect more lines
#         max_lines=15,
#         show_accumulator=True
#     )
#
#     print(f"Found {len(lines)} lines above threshold")