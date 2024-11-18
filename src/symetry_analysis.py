import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def analyze_symmetry(preprocessed_image, contours):
    """
    Analyze the symmetry of the main snowflake contour with rotational symmetry.
    """
    if not contours or len(contours) == 0:
        return []

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
    
    resized = cv2.resize(square, (100, 100))
    
    # rotational symmetry scores for multiple angles
    angles = [60, 120, 180, 240, 300] 
    symmetry_scores = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((50, 50), angle, 1.0)
        rotated = cv2.warpAffine(resized, M, (100, 100))
        score = ssim(resized, rotated)
        symmetry_scores.append(score)
    
    # circularity https://en.wikipedia.org/wiki/Roundness#:~:text=Roundness%20%3D%20%E2%81%A04π%20×,for%20highly%20non%2Dcircular%20shapes.
    perimeter = cv2.arcLength(main_contour, True)
    area = cv2.contourArea(main_contour)
    if perimeter == 0:
        print("Perimeter is 0; cannot calculate circularity")
        return []
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    # our ffinal symmetry score combines rotational symmetry and circularity
    avg_symmetry = np.mean(symmetry_scores)
    final_score = (circularity + avg_symmetry) / 2
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(resized, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("60° Rotation")
    rotated_60 = cv2.warpAffine(resized, cv2.getRotationMatrix2D((50, 50), 60, 1.0), (100, 100))
    plt.imshow(rotated_60, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("120° Rotation")
    rotated_120 = cv2.warpAffine(resized, cv2.getRotationMatrix2D((50, 50), 120, 1.0), (100, 100))
    plt.imshow(rotated_120, cmap='gray')
    plt.axis('off')
    
    plt.show()
    
    print(f"Circularity: {circularity:.2f}")
    print(f"Average Rotational Symmetry: {avg_symmetry:.2f}")
    print(f"Final Symmetry Score: {final_score:.2f}")
    
    return [final_score]