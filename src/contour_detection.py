import cv2
import numpy as np

def detect_contours(preprocessed_image):
    """
    Detect contours in the preprocessed image using Canny edge detection.
    """
    # Apply Canny edge detection
    edges = cv2.Canny(preprocessed_image, threshold1=100, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw contours
    contour_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    return contour_image, contours

def filter_contours(preprocessed_image, contours, min_area=100, max_area=5000, circularity_threshold=0.01):
    """
    Filter contours based on area and shape characteristics.
    """
    filtered_contours = []

    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue  # Skip small or very large contours

        # Calculate circularity: 4 * pi * (Area) / (Perimeter^2)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < circularity_threshold:  # Adjust threshold based on desired shapes
            continue

        filtered_contours.append(contour)

    # Visualize filtered contours
    contour_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
    cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    return contour_image, filtered_contours
