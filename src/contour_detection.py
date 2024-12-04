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

def segment_image(image_path, threshold=100):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Step 3: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    # Step 4: Extract Gabor features for each valid region
    valid_centroids = []
    gabor_features = []
    for i, (x, y) in enumerate(centroids):
        if stats[i, cv2.CC_STAT_AREA] < 5:  # Skip small regions
            continue
        # Crop a small region around the centroid
        x, y = int(x), int(y)
        patch = gray[max(0, y-10):y+10, max(0, x-10):x+10]
        if patch.size == 0:
            continue
        # Compute Gabor features
        feature, _ = gabor(patch, frequency=0.6)
        gabor_features.append(feature.mean())
        valid_centroids.append((x, y))

    # Convert valid centroids to NumPy array for easier processing
    valid_centroids = np.array(valid_centroids)

    # Merge centers based on Euclidean and feature distance
    reduced_centroids = []
    for i, c1 in enumerate(valid_centroids):
        merged = False
        for j, c2 in enumerate(reduced_centroids):
            spatial_dist = euclidean(c1, c2)
            feature_dist = abs(gabor_features[i] - gabor_features[j])
            if spatial_dist * feature_dist < threshold:
                merged = True
                break
        if not merged:
            reduced_centroids.append(c1)

    # Step 5: Apply FCM Clustering
    reduced_centroids = np.array(reduced_centroids)
    reshaped = img.reshape((-1, 3)).astype(np.float32)

    # Apply Fuzzy C-means Clustering
    cntr, u, u0, d, jm, p, fpc = cmeans(
        data=reshaped.T, c=3, m=2, error=0.005, maxiter=1000, init=None
    )

    # Assign clusters to pixels
    cluster_map = np.argmax(u, axis=0).reshape(img.shape[:2])

    # Visualize the segmented image
    segmented_image = np.zeros_like(img)
    for i in range(3):  # Assuming 3 clusters
        segmented_image[cluster_map == i] = (i * 85, i * 85, i * 85)



    """plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()"""

    return segmented_image
