from src import preprocessing as pp
from src import contour_detection as cd
from src import symetry_analysis as sa

import matplotlib.pyplot as plt
import cv2
import os
import json

def main(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = pp.preprocess_image(image)
    _, contours = cd.detect_contours(preprocessed_image)
    _, filtered_contours = cd.filter_contours(preprocessed_image, contours, min_area=100, max_area=10000)
    symmetry_score = sa.analyze_symmetry(preprocessed_image, filtered_contours, vis=False)

    return symmetry_score


if __name__ == '__main__':
    scores = {}
    data_folder = "data/"
    images = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f.lower().endswith('.jpg')]
    for path in images:
        scores[path] = main(path)
        #scores.append(main(path))
        
    output_file = "scores.json"
    with open(output_file, 'w') as f:
        json.dump(scores, f)