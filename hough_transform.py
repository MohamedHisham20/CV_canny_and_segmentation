import cv2
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from skimage.feature import canny
from matplotlib.patches import Ellipse


def compute_hough_ellipse_focus(edges, min_d=10, max_d=50, step_size=2, tolerance=1.0, min_votes=5,
                                min_foci_dist=5, max_foci_dist=None, non_max_suppression=True):
    """
    Improved Hough Transform for ellipse detection using the focus-based representation.

    Parameters:
    - edges: Binary edge image (numpy array)
    - min_d, max_d: Range of sum of distances defining the ellipse
    - step_size: Step size for reducing the search space
    - tolerance: Tolerance for considering a point on the ellipse
    - min_votes: Minimum number of votes required to consider an ellipse detection valid
    - min_foci_dist: Minimum distance between foci to be considered
    - max_foci_dist: Maximum distance between foci to be considered (if None, computed based on image size)
    - non_max_suppression: Whether to apply non-maximum suppression to remove duplicate detections

    Returns:
    - ellipses: List of detected ellipses as (f1, f2, d, votes)
    """
    # Get edge points
    edge_points = np.argwhere(edges > 0)

    # If too few edge points, return empty list
    if len(edge_points) < 3:
        return []

    # Initialize accumulator and results
    accumulator = {}
    ellipses = []

    # Set default max_foci_dist if not provided
    if max_foci_dist is None:
        max_foci_dist = min(edges.shape) / 2

    # Sample focus candidates to reduce computational complexity
    n_samples = min(100, len(edge_points))
    indices = np.random.choice(len(edge_points), n_samples, replace=False)
    focus_candidates = edge_points[indices]

    # Iterate over pairs of edge points as possible foci
    for (x1, y1), (x2, y2) in combinations(focus_candidates, 2):
        f1, f2 = np.array([x1, y1]), np.array([x2, y2])

        # Compute distance between foci
        foci_dist = np.linalg.norm(f1 - f2)

        # Skip if foci are too close or too far
        if foci_dist < min_foci_dist or foci_dist > max_foci_dist:
            continue

        # Get range of possible d values
        d_min = max(min_d, foci_dist * 1.05)  # Slightly larger than foci distance
        d_max = min(max_d, foci_dist * 2.5)  # Upper bound based on typical ellipse shapes

        # Skip if range is invalid
        if d_min >= d_max:
            continue

        # For each d value, check which edge points lie on the ellipse
        for d in np.arange(d_min, d_max, step_size):
            # Calculate distances from all edge points to both foci
            dist_to_f1 = cdist(edge_points, [f1]).reshape(-1)
            dist_to_f2 = cdist(edge_points, [f2]).reshape(-1)

            # Calculate total distance
            total_dist = dist_to_f1 + dist_to_f2

            # Find points that lie on the ellipse within tolerance
            on_ellipse = np.abs(total_dist - d) <= tolerance
            ellipse_points = edge_points[on_ellipse]
            votes = len(ellipse_points)

            # If enough points support this ellipse, add to accumulator
            if votes >= min_votes:
                # Check if the points are well distributed around the ellipse (not just in one area)
                # Convert to polar coordinates relative to ellipse center
                center = (f1 + f2) / 2
                angles = np.arctan2(ellipse_points[:, 0] - center[0],
                                    ellipse_points[:, 1] - center[1])

                # Count points in different angle bins
                bins = np.linspace(-np.pi, np.pi, 8)
                hist, _ = np.histogram(angles, bins)

                # If points are well distributed (at least 3 bins have points)
                if np.sum(hist > 0) >= 3:
                    key = (int(x1), int(y1), int(x2), int(y2), int(d))
                    accumulator[key] = (votes, ellipse_points)

    # Convert accumulator to list of ellipses with votes
    for (x1, y1, x2, y2, d), (votes, points) in accumulator.items():
        ellipses.append(((x1, y1), (x2, y2), d, votes))

    # Sort ellipses by number of votes (descending)
    ellipses.sort(key=lambda x: x[3], reverse=True)

    # Apply non-maximum suppression to remove duplicates if requested
    if non_max_suppression and ellipses:
        filtered_ellipses = [ellipses[0]]  # Keep the highest voted ellipse

        for ellipse in ellipses[1:]:
            f1, f2, d, votes = ellipse

            # Check if this ellipse is too similar to any already kept ellipse
            is_duplicate = False
            for kept_f1, kept_f2, kept_d, kept_votes in filtered_ellipses:
                # Calculate similarity based on foci and d
                f1_dist = np.linalg.norm(np.array(f1) - np.array(kept_f1))
                f2_dist = np.linalg.norm(np.array(f2) - np.array(kept_f2))
                d_diff = abs(d - kept_d)

                # If both foci are close and d is similar, consider it a duplicate
                if (f1_dist < min_foci_dist and f2_dist < min_foci_dist and d_diff < step_size * 2) or \
                        (f1_dist < min_foci_dist and d_diff < step_size * 2) or \
                        (f2_dist < min_foci_dist and d_diff < step_size * 2):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_ellipses.append(ellipse)

        return filtered_ellipses

    return ellipses


def generate_test_image(image_size=200, noise_level=0.05):
    """Generate a test image with ellipses"""
    # Create empty image
    img = np.zeros((image_size, image_size))

    # Add an ellipse
    rr, cc = ellipse(100, 100, 70, 40, img.shape)
    img[rr, cc] = 1

    # Add another ellipse
    rr, cc = ellipse(60, 150, 30, 20, img.shape)
    img[rr, cc] = 1

    # Add noise
    img = img + noise_level * np.random.randn(*img.shape)
    img = np.clip(img, 0, 1)

    return img


def ellipse_params_from_foci(f1, f2, d):
    """Convert ellipse from focus representation to center/axes representation"""
    # Center is the midpoint of the foci
    center_x = (f1[0] + f2[0]) / 2
    center_y = (f1[1] + f2[1]) / 2

    # Distance between foci
    c = np.linalg.norm(np.array(f1) - np.array(f2)) / 2

    # Semi-major axis (a) is half the sum of distances
    a = d / 2

    # Semi-minor axis (b) using the formula: b² = a² - c²
    b = np.sqrt(max(0.1, a ** 2 - c ** 2))

    # Angle of rotation (in degrees)
    dx = f2[0] - f1[0]
    dy = f2[1] - f1[1]
    angle = np.degrees(np.arctan2(dy, dx))

    return center_x, center_y, a, b, angle


def detect_and_visualize_ellipses(image, min_d=20, max_d=200, step_size=2, tolerance=1.5, min_votes=10):
    """Detect ellipses in an image and visualize results"""
    # Get edges (assuming image might already be an edge image)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # Convert to grayscale if RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # If the image is not already binary, apply edge detection
    if np.max(gray) > 1 or np.min(gray) < 0:
        gray = gray / 255.0  # Normalize to [0,1]

    if np.mean(gray) > 0.5:  # If mostly white, invert
        edges = gray < 0.5
    else:
        edges = gray > 0.5

    # Apply light blur to smooth edges if needed
    edges = edges.astype(np.uint8)
    edges = cv2.GaussianBlur(edges, (3, 3), 0) > 0

    # Detect ellipses with improved algorithm
    ellipses = compute_hough_ellipse_focus(
        edges,
        min_d=min_d,
        max_d=max_d,
        step_size=step_size,
        tolerance=tolerance,
        min_votes=min_votes,
        min_foci_dist=10,
        non_max_suppression=True
    )

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Edge image
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Image')
    axes[1].axis('off')

    # Detected ellipses
    axes[2].imshow(image, cmap='gray')
    axes[2].set_title(f'Detected Ellipses ({len(ellipses)})')

    # Draw detected ellipses
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, (f1, f2, d, votes) in enumerate(ellipses[:10]):  # Show top 10 ellipses
        cx, cy, a, b, angle = ellipse_params_from_foci(f1, f2, d)
        color = colors[i % len(colors)]

        # Draw ellipse
        ellipse_patch = Ellipse((cy, cx), 2 * b, 2 * a, angle=angle - 90,
                                fill=False, edgecolor=color, linewidth=2)
        axes[2].add_patch(ellipse_patch)

        # Add label with ellipse number and votes
        axes[2].text(cy, cx, f'{i + 1}:{votes}', color='white',
                     fontsize=9, bbox=dict(facecolor=color, alpha=0.7))

        # Optionally draw foci for debugging
        # axes[2].plot(f1[1], f1[0], 'o', color=color, markersize=4)
        # axes[2].plot(f2[1], f2[0], 'o', color=color, markersize=4)

    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

    return ellipses


# Example usage for an existing image
def test_on_image(image_path=None):
    if image_path:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
    else:
        # Or use the test image directly if it's already loaded
        # Assuming 'test_image' is your variable name
        test_image_path = "images/canna.png"
        test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        image = test_image

    # Detect ellipses with parameters tuned for this image
    detected_ellipses = detect_and_visualize_ellipses(
        image,
        min_d=30,  # Minimum distance sum for ellipse
        max_d=200,  # Maximum distance sum for ellipse
        step_size=1,  # Step size for distance values
        tolerance=1.5,  # Tolerance for points on ellipse
        min_votes=5  # Minimum points required to detect ellipse
    )

    # Print results
    print(f"Found {len(detected_ellipses)} ellipses")
    for i, (f1, f2, d, votes) in enumerate(detected_ellipses[:5]):
        print(f"Ellipse {i + 1}: Foci={f1},{f2}, Distance={d}, Votes={votes}")
        cx, cy, a, b, angle = ellipse_params_from_foci(f1, f2, d)
        print(f"  Center=({cx:.1f},{cy:.1f}), Axes=({a:.1f},{b:.1f}), Angle={angle:.1f}°")
        print(f"  Eccentricity={np.linalg.norm(np.array(f1) - np.array(f2)) / d:.2f}")

# To use this with your existing image:
test_on_image()