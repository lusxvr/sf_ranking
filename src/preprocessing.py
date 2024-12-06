import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from scipy.spatial.distance import euclidean
from skfuzzy import cmeans

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

def compute_color_frequencies(image):
    """Compute unique color frequencies in the image."""
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    # Convert to float32 for FCM
    pixels = pixels.astype(np.float32)

    # Get unique colors
    unique_colors = np.unique(pixels, axis=0)

    # Optionally, reduce to main colors (e.g., using k-means or a simple heuristic)
    # For simplicity, we can just return the unique colors
    return unique_colors

def segment_image(image_path, threshold=100, verbose=False):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Visualization: Original Grayscale Image
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Grayscale Image")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    # Step 2: Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Visualization: Canny Edge Detection Result
    plt.subplot(2, 3, 2)
    plt.title("Canny Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    # Step 3: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

    # Visualization: Connected Components
    plt.subplot(2, 3, 3)
    plt.title("Connected Components")
    plt.imshow(labels, cmap='nipy_spectral')
    plt.axis('off')

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

    # Visualization: Reduced Centroids
    plt.subplot(2, 3, 4)
    plt.title("Reduced Centroids")
    centroid_image = np.zeros_like(gray)  # Create a blank image
    for (x, y) in reduced_centroids:
        if 0 <= x < centroid_image.shape[1] and 0 <= y < centroid_image.shape[0]:  # Check bounds
            cv2.circle(centroid_image, (x, y), 5, (255, 255, 255), -1)  # Draw white circles
    plt.imshow(centroid_image, cmap='gray')
    plt.axis('off')

    # Step 5: Compute Color Frequencies
    unique_colors = compute_color_frequencies(img)
    print(f"Unique colors: {len(unique_colors)}")

    # Apply FCM Clustering using the unique colors
    reshaped = unique_colors.reshape((-1, 3)).astype(np.float32)

    # Apply Fuzzy C-means Clustering with the number of unique colors
    c = len(unique_colors)  # Number of clusters based on unique colors
    cntr, u, u0, d, jm, p, fpc = cmeans(
        data=reshaped.T, c=c, m=2, error=0.005, maxiter=1000, init=None
    )

    # Assign clusters to pixels
    cluster_map = np.argmax(u, axis=0).reshape(img.shape[:2])
    print(np.unique(cluster_map))

    # Create the segmented image
    segmented_image = np.zeros_like(img)

    # Assign colors based on the cluster map
    for i in range(c):  # Use the number of unique colors
        color = unique_colors[i]
        segmented_image[cluster_map == i] = color  # Assign the main color

    # Visualization: Final Segmented Image
    plt.subplot(2, 3, 5)
    plt.title("Final Segmented Image")
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return segmented_image