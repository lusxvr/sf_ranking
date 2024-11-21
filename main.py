from src import preprocessing as pp
from src import contour_detection as cd
from src import symetry_analysis as sa

import matplotlib.pyplot as plt
import cv2

def main(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = pp.preprocess_image(image)
    _, contours = cd.detect_contours(preprocessed_image)
    _, filtered_contours = cd.filter_contours(preprocessed_image, contours, min_area=100, max_area=10000)
    symmetry_score = sa.analyze_symmetry(preprocessed_image, filtered_contours, vis=False)

    return symmetry_score


if __name__ == '__main__':
    images = ["data/example1.jpg", "data/example2.jpg", "data/example3.jpg", "data/example4.jpg", "data/example5.jpg", "data/artificial2.png"]
    scores = []
    for path in images:
        scores.append(main(path))
    print(scores)