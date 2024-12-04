from src import preprocessing as pp
from src import contour_detection as cd
from src import symetry_analysis as sa

import matplotlib.pyplot as plt
import cv2
import os
import json

def main(image_path):
    # Read and preprocess the image
    image = pp.segment_image(image_path)
    preprocessed_image = pp.preprocess_image(image)
    contour_image, contours = cd.detect_contours(preprocessed_image)
    
    # Analyze symmetry
    symmetry_score = sa.analyze_symmetry(preprocessed_image, contours, vis=False)

    return image, contour_image, symmetry_score  # Return image, contour image, and symmetry score

if __name__ == '__main__':
    scores = {}
    images = []
    contours = []
    data_folder = "data/"
    image_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.lower().endswith('.jpg')]
    
    for path in image_files:
        image, contour_image, symmetry_score = main(path)
        scores[path] = symmetry_score
        images.append((image, symmetry_score))  # Store the image and its score
        contours.append(contour_image)  # Store the contour image

    # Plot all images with their symmetry scores and contours
    num_images = len(images)
    cols = 4  # Set the number of columns
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    plt.figure(figsize=(15, 5 * rows))  # Adjust the figure size based on the number of rows
    for i, ((img, score), contour_img) in enumerate(zip(images, contours)):
        # Plot original image with symmetry score
        plt.subplot(rows, cols, i + 1)  # Original images
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Symmetry Score: {score:.2f}')
        plt.axis('off')

        # Plot contour image
        plt.subplot(rows, cols, i + 1 + cols * rows)  # Contour images
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Contours')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save scores to JSON
    output_file = "scores_segmented.json"
    with open(output_file, 'w') as f:
        json.dump(scores, f)