import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def analyze_symmetry(preprocessed_image, contours):
    """
    Analyze the symmetry of detected contours using circularity metric and rotational symmetry.
    """
    symmetry_scores = []

    for contour in contours:
        # Calculate circularity: 4 * pi * (Area) / (Perimeter^2)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0:
            continue  # Skip degenerate contours

        circularity = 4 * np.pi * (area / (perimeter ** 2))

        # Create bounding box to center the contour for rotational symmetry
        x, y, w, h = cv2.boundingRect(contour)
        cropped = preprocessed_image[y:y+h, x:x+w]

        # Resize for symmetry comparison
        resized = cv2.resize(cropped, (100, 100))

        # Rotational symmetry (compare with 180-degree rotated image)
        rotated = cv2.rotate(resized, cv2.ROTATE_180)
        rotational_symmetry = ssim(resized, rotated)

        # Combine metrics
        symmetry_score = (circularity + rotational_symmetry) / 2
        symmetry_scores.append(symmetry_score)

        # Display individual results for debugging
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(resized, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Rotated")
        plt.imshow(rotated, cmap='gray')
        plt.axis('off')

        plt.show()

        print(f"Symmetry Score: {symmetry_score:.2f}")

    return symmetry_scores