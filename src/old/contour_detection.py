import cv2
import numpy as np
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import gabor
from scipy.spatial.distance import euclidean
from skfuzzy import cmeans
import matplotlib.pyplot as plt




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
    #kernel = np.ones((3,3), np.uint8)
    #connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #final_contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_contour = max(contours, key=cv2.contourArea)

    # Added smoothing of the contour. This helps for real world snowflakes but doesnt for artificial ones because they already have a perfect contour
    epsilon = 0.0001 * cv2.arcLength(main_contour, True)
    smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    contour_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, [smoothed_contour], -1, (0, 255, 0), 2)

    return contour_image, [smoothed_contour]


