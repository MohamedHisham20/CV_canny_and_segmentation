import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

class PostContour:
    def __init__(self, data):
        self.path = data

    @staticmethod
    def calculate_area(path):
        area = 0
        for i in range(len(path)):
            x1, y1 = path[i]
            x2, y2 = path[(i + 1) % len(path)] 
            area += (x1 * y2 - x2 * y1)
        return 0.5 * abs(area)

    @staticmethod
    def calculate_perimeter(path):
        perimeter = 0
        for i in range(len(path)):
            p1 = path[i]
            p2 = path[(i + 1) % len(path)]
            perimeter += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return perimeter

    @staticmethod
    def generate_chain_code(path):
        direction_map = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    
        chain_code = []
    
        for i in range(1, len(path)):
            # Get movement direction
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            
            dx = np.clip(round(dx), -1, 1)
            dy = np.clip(round(dy), -1, 1)

            chain_code.append(direction_map[(dx, dy)])

        return chain_code
    
    


#######TESTING#######

# image = np.zeros((100, 100), dtype=np.uint8)
# cv2.circle(image, (50, 50), 30, 255, -1)  # Draw filled circle
# image = gaussian(image, sigma=1)  # Smooth image

# # Create initial snake (circle around the object)
# t = np.linspace(0, 2 * np.pi, 100)
# x = 50 + 35 * np.cos(t)  # Slightly larger than the object
# y = 50 + 35 * np.sin(t)
# init_snake = np.array([x, y]).T

# # Apply active contour model
# snake = active_contour(image, init_snake, alpha=0.01, beta=0.1, gamma=0.01)

# # Round snake points to nearest pixels
# contour_path = [(int(round(pt[0])), int(round(pt[1]))) for pt in snake]

# # Compute Chain Code
# chain_code = PostContour.generate_chain_code(contour_path)

# # Display results
# plt.figure(figsize=(6, 6))
# plt.imshow(image, cmap='gray')
# plt.plot(init_snake[:, 0], init_snake[:, 1], '--r', label='Initial Snake')
# plt.plot(snake[:, 0], snake[:, 1], '-b', label='Final Contour')
# plt.scatter([p[0] for p in contour_path], [p[1] for p in contour_path], c='yellow', s=5)
# plt.legend()
# plt.title("Active Contour on Image")
# plt.show()

# # Print Chain Code
# print("Chain Code:", chain_code)
# # **Testing**
# post_contour = PostContour([])

# print("Area:", post_contour.calculate_area(post_contour.path))
# print("Perimeter:", post_contour.calculate_perimeter(post_contour.path))
# print("Chain Code:", post_contour.generate_chain_code(post_contour.path))
