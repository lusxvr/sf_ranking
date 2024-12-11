import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

def calculate_corr(scores_path, votes_path):
    # Load JSON data
    with open(scores_path, 'r') as f:
        scores_data = json.load(f)

    with open(votes_path, 'r') as f:
        votes_data = json.load(f)

    # Ensure data alignment
    image_ids = list(scores_data.keys())
    scores = []
    votes = []

    for img_id in image_ids:
        if img_id in votes_data:
            scores.append(scores_data[img_id])
            votes.append(votes_data[img_id])

    # Calculate correlation
    spearman_corr, p_value = spearmanr(scores, votes)

    # Print results
    print(f"Spearman correlation: {spearman_corr}")
    print(f"P-value: {p_value}")

    # Fit a linear regression line for the trendline
    slope, intercept, r_value, _, _ = linregress(scores, votes)
    trendline = [slope * x + intercept for x in scores]

    # Visualize the data
    plt.figure(figsize=(8, 6))
    plt.scatter(scores, votes, alpha=0.7, label='Data points')
    plt.plot(scores, trendline, color='red', label='Trendline (Linear Fit)')
    plt.xlabel('Scores')
    plt.ylabel('Votes')
    plt.title('Alignment of Scores and Votes')
    plt.legend()
    plt.show()

    # Interpretation
    if p_value < 0.05:
        print("The metrics show a statistically significant alignment.")
    else:
        print("The metrics do not show a statistically significant alignment.")