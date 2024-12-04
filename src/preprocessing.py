import cv2
import cv2
import numpy as np
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import gabor
from scipy.spatial.distance import euclidean
from skfuzzy import cmeans
import matplotlib.pyplot as plt

# https://f1000research.com/articles/12-1312
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
        1
    )
    # Apply Otsu's thresholding
    # _, thresh = cv2.threshold(
    #     blurred,
    #     0,
    #     255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    return thresh

def segment_image(image_path, threshold=100, verbose=False):
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

    # Step 5: Apply FCM Clustering with c=2 to enforce two clusters
    reduced_centroids = np.array(reduced_centroids)
    reshaped = img.reshape((-1, 3)).astype(np.float32)

    # Apply Fuzzy C-means Clustering with c=2
    cntr, u, u0, d, jm, p, fpc = cmeans(
        data=reshaped.T, c=2, m=2, error=0.005, maxiter=1000, init=None
    )

    # Assign clusters to pixels
    cluster_map = np.argmax(u, axis=0).reshape(img.shape[:2])
    print(np.unique(cluster_map))

    # Create the segmented image
    segmented_image = np.zeros_like(img)

    # Assign colors: one cluster to white and the other to black
    for i in range(2):  # Since we have enforced c=2
        if i == 0:
            segmented_image[cluster_map == i] = (255, 255, 255)  # White
        else:
            segmented_image[cluster_map == i] = (0, 0, 0)  # Black

    if verbose:
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Segmented Image")
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return segmented_image