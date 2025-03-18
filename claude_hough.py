import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from scipy import ndimage
import random


# First, include the ellipse_detection function we defined earlier
def ellipse_detection(img, params=None, verbose=True):
    """
    Memory-efficient ellipse detection implementation.

    Parameters:
    -----------
    img : ndarray
        One-channel input image (grayscale or binary).
    params : dict, optional
        Parameters of the algorithm.
    verbose : bool, optional
        Whether to print intermediate log messages

    Returns:
    --------
    ndarray
        Matrix of best fits. Each row contains:
        [x0, y0, a, b, alpha, score] representing the center of the ellipse,
        its major and minor semiaxis, its angle in degrees, and score.
    """
    # Default values
    if params is None:
        params = {}

    # Parameters to constrain the search
    minMajorAxis = params.get('minMajorAxis', 10)
    maxMajorAxis = params.get('maxMajorAxis', 200)
    rotation = params.get('rotation', 0)
    rotationSpan = params.get('rotationSpan', 0)
    minAspectRatio = params.get('minAspectRatio', 0.1)
    randomize = params.get('randomize', 2)

    # Others
    numBest = params.get('numBest', 3)
    uniformWeights = params.get('uniformWeights', True)
    smoothStddev = params.get('smoothStddev', 1)

    eps = 0.0001
    bestFits = np.zeros((numBest, 6))
    rotationSpan = min(rotationSpan, 90)

    # Create Gaussian kernel for smoothing
    kernel_size = max(1, round(smoothStddev * 6))
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    H = np.exp(-(x ** 2) / (2 * smoothStddev ** 2))
    H = H / np.sum(H)  # Normalize

    # Find non-zero points
    Y, X = np.nonzero(img)
    Y = Y.astype(np.float32)
    X = X.astype(np.float32)
    N = len(Y)

    if verbose:
        print(f'Total non-zero points: {N}')

    # If we have too many points, subsample
    max_points = params.get('max_points', 5000)  # New parameter for point limitation
    if N > max_points:
        if verbose:
            print(f'Too many points, subsampling to {max_points}')
        indices = np.random.choice(N, max_points, replace=False)
        X = X[indices]
        Y = Y[indices]
        N = max_points

    # Generate pairs of points differently to avoid full matrix computation
    # Instead of computing all-to-all distances, we'll sample random pairs
    num_pairs = min(N * (N - 1) // 2, N * params.get('randomize', 2))

    if verbose:
        print(f'Processing {num_pairs} point pairs')

    # Pre-allocate arrays for valid pairs
    valid_pairs = []

    # Process in batches to avoid memory issues
    batch_size = 10000
    num_batches = (num_pairs + batch_size - 1) // batch_size

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, num_pairs)
        batch_count = end_idx - start_idx

        # Generate random pairs for this batch
        if params.get('randomize', 2) > 0:
            i_indices = np.random.randint(0, N, batch_count)
            j_indices = np.random.randint(0, N, batch_count)

            # Ensure i != j
            same_idx = i_indices == j_indices
            j_indices[same_idx] = (j_indices[same_idx] + 1) % N
        else:
            # If randomize is disabled, generate sequential pairs systematically
            pairs_per_point = max(1, (batch_count + N - 1) // N)
            i_indices = np.repeat(np.arange(N), pairs_per_point)[:batch_count]
            j_indices = (i_indices + np.random.randint(1, N, batch_count)) % N

        # Calculate distances
        dx = X[i_indices] - X[j_indices]
        dy = Y[i_indices] - Y[j_indices]
        distances_sq = dx ** 2 + dy ** 2

        # Filter by distance constraint
        valid_dist = (distances_sq >= minMajorAxis ** 2) & (distances_sq <= maxMajorAxis ** 2)

        # Filter by angle constraint if needed
        if rotationSpan > 0:
            tangents = dy[valid_dist] / (dx[valid_dist] + eps)
            tanLo = np.tan(np.radians(rotation - rotationSpan))
            tanHi = np.tan(np.radians(rotation + rotationSpan))

            if tanLo < tanHi:
                valid_angle = (tangents > tanLo) & (tangents < tanHi)
            else:
                valid_angle = (tangents > tanLo) | (tangents < tanHi)

            # Get indices of valid pairs
            i_valid = i_indices[valid_dist][valid_angle]
            j_valid = j_indices[valid_dist][valid_angle]
            d_valid = distances_sq[valid_dist][valid_angle]
        else:
            # Get indices of valid pairs
            i_valid = i_indices[valid_dist]
            j_valid = j_indices[valid_dist]
            d_valid = distances_sq[valid_dist]

        # Store valid pairs
        for idx in range(len(i_valid)):
            valid_pairs.append((i_valid[idx], j_valid[idx], d_valid[idx]))

    if verbose:
        print(f'Valid pairs after filtering: {len(valid_pairs)}')

    # Shuffle pairs to ensure randomness
    random.shuffle(valid_pairs)

    # Limit the number of pairs to process
    max_pairs_to_process = min(len(valid_pairs), N * params.get('randomize', 2))
    valid_pairs = valid_pairs[:max_pairs_to_process]

    if verbose:
        print(f'Processing {len(valid_pairs)} filtered pairs')

    # Process valid pairs
    for i, j, dist_sq in valid_pairs:
        x1, y1 = X[i], Y[i]
        x2, y2 = X[j], Y[j]

        # Compute center & major axis
        x0, y0 = (x1 + x2) / 2, (y1 + y2) / 2
        aSq = dist_sq / 4

        # For each point, compute distance to center
        thirdPtDistsSq = (X - x0) ** 2 + (Y - y0) ** 2
        K = thirdPtDistsSq <= aSq  # Points within major axis distance

        if np.sum(K) == 0:
            continue

        # Get minor axis propositions for all other points
        fSq = (X[K] - x2) ** 2 + (Y[K] - y2) ** 2
        cosTau = (aSq + thirdPtDistsSq[K] - fSq) / (2 * np.sqrt(aSq * thirdPtDistsSq[K] + eps))
        cosTau = np.clip(cosTau, -1, 1)  # Handle floating point issues
        sinTauSq = 1 - cosTau ** 2

        # Calculate minor axis
        denominator = aSq - thirdPtDistsSq[K] * cosTau ** 2 + eps
        b = np.sqrt((aSq * thirdPtDistsSq[K] * sinTauSq) / denominator)

        # Create accumulator
        idxs = np.ceil(b + eps).astype(int)
        idxs = np.clip(idxs, 0, maxMajorAxis - 1)

        if uniformWeights:
            weights = np.ones_like(idxs)
        else:
            weights = img[Y[K].astype(int), X[K].astype(int)]

        accumulator = np.zeros(maxMajorAxis)
        np.add.at(accumulator, idxs, weights)

        # Smooth accumulator and find most busy bin
        accumulator = ndimage.convolve1d(accumulator, H, mode='reflect')
        min_aspect_cutoff = int(np.ceil(np.sqrt(aSq) * minAspectRatio))
        accumulator[:min_aspect_cutoff] = 0

        score = np.max(accumulator)
        idx = np.argmax(accumulator)

        # Keep only the numBest best hypotheses
        if bestFits[-1, -1] < score:
            angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
            bestFits[-1, :] = [x0, y0, np.sqrt(aSq), idx, angle, score]
            if numBest > 1:
                bestFits = bestFits[bestFits[:, 5].argsort()[::-1]]

    return bestFits

# Function to create a test image with multiple ellipses and noise
def create_test_image(width=300, height=200, num_ellipses=3, noise_level=0.05):
    """Create a test image with multiple ellipses and noise"""
    image = np.zeros((height, width))

    # Create random ellipses
    for _ in range(num_ellipses):
        # Random ellipse parameters
        center_x = random.randint(50, width - 50)
        center_y = random.randint(50, height - 50)
        a = random.randint(20, 60)  # Major axis
        b = random.randint(10, a)  # Minor axis (smaller than major)
        angle = random.randint(0, 180)  # Rotation angle in degrees

        # Generate ellipse points
        rr, cc = draw.ellipse(center_y, center_x, b, a, shape=image.shape, rotation=np.radians(angle))
        image[rr, cc] = 1

    # # Add noise
    # if noise_level > 0:
    #     noise = np.random.random(image.shape) < noise_level
    #     image = np.logical_or(image, noise).astype(np.uint8)

    # #save the image
    # cv2.imwrite("images/maskyy.png", image*255)

    return image


from skimage.draw import ellipse, ellipse_perimeter
import numpy as np

def draw_ellipses(image, ellipses, colors=None):
    """Draw detected ellipses on a binary or grayscale image, filling their interior."""
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Convert grayscale/binary image to 3-channel grayscale
    if image.ndim == 2:
        rgb_image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel grayscale
    else:
        rgb_image = image.copy()

    for i, ellipse_params in enumerate(ellipses):
        if ellipse_params[5] == 0:  # Skip ellipses with zero score
            continue

        color = colors[i % len(colors)]  # Pick a color
        x0, y0, a, b, angle = ellipse_params[:5]

        # Draw filled ellipse
        rr_fill, cc_fill = ellipse(
            int(y0), int(x0), int(b), int(a),
            rotation=np.radians(angle),
            shape=image.shape
        )
        rgb_image[rr_fill, cc_fill] = color  # Fill the ellipse interior

        # Draw ellipse perimeter
        rr_perim, cc_perim = ellipse_perimeter(
            int(y0), int(x0), int(b), int(a),
            orientation=np.radians(angle),
            shape=image.shape
        )
        valid_idx = (0 <= rr_perim) & (rr_perim < image.shape[0]) & (0 <= cc_perim) & (cc_perim < image.shape[1])
        rr_perim, cc_perim = rr_perim[valid_idx], cc_perim[valid_idx]

        rgb_image[rr_perim, cc_perim] = color  # Draw perimeter

    return rgb_image


# Main execution example
def main():
    # 1. Generate a test image or load your own image
    # Option 1: Generate a test image with ellipses
    test_image = create_test_image(width=400, height=300, num_ellipses=3, noise_level=0.02)

    # Option 2: Load and process your own image
    # image_path = "your_image.jpg"
    # original_image = io.imread(image_path)
    # if len(original_image.shape) > 2:  # Convert to grayscale if colored
    #     gray_image = color.rgb2gray(original_image)
    # else:
    #     gray_image = original_image
    #
    # # Apply edge detection to get boundary points
    # edges = feature.canny(gray_image, sigma=2)
    # test_image = edges

    # 2. Set up detection parameters
    params = {
        'minMajorAxis': 5,
        'maxMajorAxis': 10000,
        'rotation': -360,
        'rotationSpan': 360,
        'minAspectRatio': 0.3,
        'randomize': 5,
        'numBest': 10,
        'uniformWeights': True,
        'smoothStddev': 1,
        'max_points': 8000  # Limit the number of points to process
    }

    test_image_path = "images/canna.png"
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    # 3. Run ellipse detection
    ellipses = ellipse_detection(test_image, params, verbose=True)

    # 4. Print detected ellipses
    print("\nDetected Ellipses:")
    print("Center (x,y) | Major | Minor | Angle | Score")
    print("-" * 50)
    for i, ellipse in enumerate(ellipses):
        if ellipse[5] > 0:  # Only print valid ellipses
            print(
                f"Ellipse {i + 1}: ({ellipse[0]:.1f}, {ellipse[1]:.1f}) | {ellipse[2]:.1f} | {ellipse[3]:.1f} | {ellipse[4]:.1f}° | {ellipse[5]:.1f}")

    # 5. Visualize results
    result_image = draw_ellipses(test_image, ellipses)

    # Display original and result
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title('Detected Ellipses')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Run the example
if __name__ == "__main__":
    main()