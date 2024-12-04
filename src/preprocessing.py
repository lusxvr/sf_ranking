import cv2

def preprocess_image(image):
    """
    Simple grayscale conversion only
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray_image, 5, 75, 75)
    thresh = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    # Apply Otsu's thresholding
    # _, thresh = cv2.threshold(
    #     blurred,
    #     0,
    #     255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    return thresh