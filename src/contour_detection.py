import cv2
import numpy as np

def detect_contours(preprocessed_image):
    """
    Detect and combine all snowflake contours into one connected contour
    """
   
    #mask from all contours
    mask = np.zeros_like(preprocessed_image)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # connect components using morphological operations https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # todo this is sometimes too rough  
    kernel = np.ones((3,3), np.uint8)
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    final_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_contour = max(final_contours, key=cv2.contourArea)

    # Added smoothing of the contour. This helps for real world snowflakes but doesnt for artificial ones because they already have a perfect contour
    epsilon = 0.0001 * cv2.arcLength(main_contour, True)
    smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    contour_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [smoothed_contour], -1, (0, 255, 0), 2)

    return contour_image, [smoothed_contour]

def filter_contours(preprocessed_image, contours, min_area=1000, max_area=100000, circularity_threshold=0.2):
    """
    Verify and clean up the main snowflake contour.
    Returns None if the contour doesn't meet the criteria for a valid snowflake.
    """
    if not contours or len(contours) == 0:
        return None, []

    main_contour = contours[0]
    
    # #Calculate area
    # area = cv2.contourArea(main_contour)
    # if area < min_area or area > max_area:
    #     print("POT. DEPRECEATED: Contour is too small or too large; contour area: ", area)
    #     #return None, []  # Contour is too small or too large

    # # circularity
    # perimeter = cv2.arcLength(main_contour, True)
    # if perimeter == 0:
    #     print("Contour has no perimeter; perimeter: ", perimeter)
        
    # circularity = 4 * np.pi * (area / (perimeter ** 2))
    # if circularity < circularity_threshold:
    #     print("POT. DEPRECEATED: Contour is not circular enough to be a snowflake; circularity: ", circularity)

    contour_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [main_contour], -1, (0, 255, 0), 2)
    return contour_image, [main_contour]
