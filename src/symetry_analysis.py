import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import rotate

def normalized_cross_correlation(image1, image2):
    """Computes Normalized Cross-Correlation (NCC) between two images."""
    image1_flat = image1.flatten().astype(np.float32)
    image2_flat = image2.flatten().astype(np.float32)
    mean1, mean2 = np.mean(image1_flat), np.mean(image2_flat)
    numerator = np.sum((image1_flat - mean1) * (image2_flat - mean2))
    denominator = np.sqrt(np.sum((image1_flat - mean1) ** 2) * np.sum((image2_flat - mean2) ** 2))
    return numerator / denominator

def analyze_symmetry(preprocessed_image, contours, vis=False):
    """
    Analyze the symmetry of the main snowflake contour with rotational symmetry.
    """
    #if not contours or len(contours) == 0:
    #    return []

    assert len(contours) == 1, "Expected exactly one contour"
    main_contour = contours[0]
    
    mask = np.zeros_like(preprocessed_image)
    cv2.drawContours(mask, [main_contour], -1, (255), thickness=cv2.FILLED)
    
    #bounding box of contour for cropping
    x, y, w, h = cv2.boundingRect(main_contour)
    masked = cv2.bitwise_and(preprocessed_image, mask)
    cropped = masked[y:y+h, x:x+w]
    
    # imge needs to be square; mif not, apply padding
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    square[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
    
    resized = cv2.resize(square, (512, 512))

    # Use blurred image to check rotational symetry, this smoothed the effect of artifacts
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    
    # rotational symmetry scores for multiple angles
    angles = [60, 120, 180, 240, 300] 
    symmetry_scores = []
    rotated_img = []

    for angle in angles:
        rotated = rotate(blurred, angle, reshape=False, mode='constant', cval=0, order=3)
        rotated_img.append(rotated)
        # score = ssim(blurred, rotated, data_range=255)
        score = normalized_cross_correlation(blurred, rotated)
        symmetry_scores.append(score)



    ### How pretty is a snowflake? To answer this we combine various metrics

    # Calculate the minimum enclosing circle
    ((cx, cy), radius) = cv2.minEnclosingCircle(main_contour)

    # Circularity ratio: how much of the rectangle the circle spans
    # Measure the "fit" of the bounding circle inside the bounding rectangle
    circle_diameter = 2 * radius
    rect_diagonal = np.sqrt(w**2 + h**2)
    circularity_ratio = circle_diameter / rect_diagonal

    # Measure the ratio of average rectange axis to circle diameter
    avg_rect_axis = (w+h) / 2
    axis_ratio = avg_rect_axis/circle_diameter

    # our ffinal symmetry score combines rotational symmetry and circularity
    avg_symmetry = np.mean(symmetry_scores)
    final_score = (circularity_ratio + axis_ratio + avg_symmetry) / 3

    

    if vis:
        print(f"Axis Ratio: {axis_ratio:.4f}")
        print(f"Circle/Rectangle Ratio: {circularity_ratio:.4f}")
        print(f"Average Rotational Symmetry: {avg_symmetry:.4f}")
        print(f"Final Symmetry Score: {final_score:.4f}")

        # Convert the grayscale image to BGR (3 channels) for visualization
        visualization_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

        # Add padding to the image
        padding = 60
        visualization_image = cv2.copyMakeBorder(
            visualization_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # Offset the contour to match the padded image
        main_contour_padded = main_contour + padding

        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour_padded)
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        # Calculate the minimum enclosing circle
        ((cx, cy), radius) = cv2.minEnclosingCircle(main_contour_padded)
        cv2.circle(visualization_image, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)  # Green circle

        # Add a legend
        legend_image = np.zeros((70, visualization_image.shape[1], 3), dtype=np.uint8)
        legend_image += 255
        cv2.putText(legend_image, "Bounding Rectangle (Blue)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(legend_image, "Bounding Circle (Green)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Combine the visualization image and legend
        final_visualization = np.vstack((visualization_image, legend_image))

        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 3, 1)
        plt.title("Original + Circularity")
        plt.imshow(cv2.cvtColor(final_visualization, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.title("60° Rotation")
        plt.imshow(rotated_img[0], cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.title("120° Rotation")
        plt.imshow(rotated_img[1], cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title("180° Rotation")
        plt.imshow(rotated_img[2], cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title("240° Rotation")
        plt.imshow(rotated_img[3], cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title("300° Rotation")
        plt.imshow(rotated_img[4], cmap='gray')
        plt.axis('off')
        
        plt.show()
    
    return final_score


def plot_images(images, titles, cmaps):
    plt.figure(figsize=(15, 5))
    for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def pipeline(image): 
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Enhance Contrast using Histogram Equalization
    equalized = gray  # Skip histogram equalization for simplicity

    # Step 2: Apply Canny Edge Detection
    edges = cv2.Canny(equalized, 50, 150)

    # Step 3: Fill the Object (Flood Fill on Edges)
    flood_fill = edges.copy()
    h, w = flood_fill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Perform flood fill from the edges to detect connected regions
    cv2.floodFill(flood_fill, mask, (0, 0), 255)

    # Step 4: Invert the Flood-Filled Image
    flood_fill_inverted = cv2.bitwise_not(flood_fill)

    # Step 5: Combine the Inverted Flood-Fill and Original Edges
    combined = cv2.bitwise_or(edges, flood_fill_inverted)

    # Step 6: Morphological Operations to Close Small Gaps
    kernel = np.ones((3, 3), np.uint8)
    segmentation_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 7: Invert the Final Mask to make Foreground Black and Background White
    segmentation_result = cv2.bitwise_not(segmentation_mask)
    #segmentation_result = segmentation_mask
    # Ensure binary output
    segmentation_result = (segmentation_result > 0).astype(np.uint8) * 255

    # Plot the results (optional for debugging)
    plot_images(
        [image, gray, edges, segmentation_result],
        ["Original Image", "Grayscale Image", "Canny Edges", "Segmented Image (Foreground: Black, Background: White)"],
        [None, "gray", "gray", "gray"]
    )

    return segmentation_result

def extract_contour(segmented_image, vis=True):
    """
    Extract the largest contour from the segmented image.
    Optionally visualize the contour and bounding rectangle.
    """
    # Invert the mask if the background is black and the foreground is white
    inverted = cv2.bitwise_not(segmented_image)

    # Find contours
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure at least one contour exists
    if not contours:
        raise ValueError("No contours found in the segmented image.")

    # Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    if vis:
        # Create a copy of the segmented image for visualization
        visualization_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

        # Draw the largest contour in green
        cv2.drawContours(visualization_image, [largest_contour], -1, (0, 255, 0), thickness=2)

        # Calculate and draw the bounding rectangle in blue
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Plot the visualization
        plt.figure(figsize=(6, 6))
        plt.title("Largest Contour and Bounding Rectangle")
        plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Return the largest contour as a list
    return [largest_contour]