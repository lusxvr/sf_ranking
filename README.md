<span align="center">
<h1> Snowflake Beauty Rating using Contour Detection and Symmetry Properties</h1>

<a> Luis Wiedmann, Anna Ketteler </a>

Snowflakes are celebrated for their intricate symmetry and are often seen as symbols of natural beauty. This work introduces a "Snowflake Beauty Rating Tool" to analyze and rate their aesthetic appeal. The method addresses the challenging task of segmenting objects with complex structures, such as snowflakes with their numerous branches and interrupted, 'holey' structure. Instead of relying on deep learning, we consciously chose traditional computer vision techniques to explore their capabilities in scenarios with limited data availability. Our preprocessing pipeline isolates the snowflake from the background using edge detection, flood filling, and morphological operations. Extracted contours are then analyzed to compute geometric and symmetry-based metrics, which are combined into a comprehensive beauty score that quantifies a snowflake's aesthetic properties. The method highlights the potential of traditional computer vision to analyze intricate natural patterns when data is too limited to finetune a foundation segmentation model.

</span>

## Setup

### Environment
```python
conda env create --file env.yml
conda activate cv
```

### Data
Feel free to upload your own data, for examples and the pictures we used in the development, see `data/`.

## Running the Rating Tool
To try the method, simply run `dev.ipynb`, this takes all pictures in the `data/` folder and runs our method on them and checks the correlation with the votes generated through the user study.

## Running the Webserver for Voting