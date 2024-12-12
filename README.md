<span align="center">
<h1> Geometric Beauty Rating using Contour Detection and Symmetry Properties</h1>

<a> Luis Wiedmann, Anna Ketteler </a>

Nature often exhibits a preference for geometric and symmetric patterns, with snowflakes standing out as iconic examples of intricate natural beauty. This work introduces a "Geometric Beauty Rating Tool" to analyze and rate their aesthetic appeal. The method addresses the challenging task of segmenting objects with complex structures, such as snowflakes with their numerous branches and interrupted, 'holey' structure. Instead of relying on deep learning, we consciously chose traditional computer vision techniques to explore their capabilities in scenarios with limited data availability. Our preprocessing pipeline isolates the snowflake from the background using edge detection, flood filling, and morphological operations. Extracted contours are then analyzed to compute geometric and symmetry-based metrics, which are combined into a comprehensive beauty score that quantifies a snowflake's aesthetic properties. The method highlights the potential of traditional computer vision to analyze intricate natural patterns when data is too limited to finetune a foundation segmentation model.

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
To try the method, simply run `dev.ipynb`. This notebook applies our method to all images in the `data/` folder, and checks the correlation of the computed scores with the votes generated through the user study.

## Running the Webapplication for Voting
To explore our user study, the webapplication used to collect votes on the snowflakes' beauty can be run locally. After installing Node.js and npm on your machine, go to the top level directory of this project and run:
```bash
npm install
node server.js
```
Then, the server is running on http://localhost:3000
Afterbeing presented 15 random pairs of snowflake images, the localhost tab will close automatically.