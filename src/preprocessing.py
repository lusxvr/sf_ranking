import cv2
import numpy as np

def preprocess_image(image):
    """
    Load and preprocess the image: grayscale conversion
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image