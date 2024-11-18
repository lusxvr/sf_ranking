import cv2

def preprocess_image(image):
    """
    Simple grayscale conversion only
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image