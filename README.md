# Satellite Image Classification Project

This project is my first machine learning assignment for the university course. It focuses on classifying land cover types in satellite images using unsupervised and supervised learning techniques.

## Project Overview

The project performs the following tasks:
1. Loads satellite images
2. Uses K-means clustering for unsupervised classification
3. Applies supervised classification using Random Forest (as a Maximum Likelihood classifier) and Naive Bayes
4. Visualizes the results of both unsupervised and supervised classifications

## Dependencies

To run this project, you need the following Python libraries:
- numpy
- rasterio
- scikit-learn
- matplotlib

You can install these dependencies using pip:

```
pip install numpy rasterio scikit-learn matplotlib
```

## Dataset

The project uses satellite images stored in the `Data/` directory. Ensure you have the following image files in this directory:
- 1.jpg
- 2.jpg
- 3.jpg
- 4.jpg
- 5.jpg
- 6.jpg
- 7.jpg

## How to Run

1. Ensure all dependencies are installed and the dataset is in place.
2. Run the script:
   ```
   python satellite_image_classification.py
   ```
3. The script will output:
   - A visualization of K-means clustering results
   - Accuracy and classification reports for Random Forest and Naive Bayes classifiers
   - Visualizations of predictions from both classifiers

## Project Structure

The project consists of several main components:

1. Data Loading: Uses `rasterio` to load satellite images.
2. Data Preparation: Reshapes the image data for machine learning algorithms.
3. Unsupervised Learning: Applies K-means clustering to identify land cover classes.
4. Supervised Learning: 
   - Splits data into training and testing sets.
   - Trains and evaluates a Random Forest classifier (as Maximum Likelihood classifier).
   - Trains and evaluates a Naive Bayes classifier.
5. Visualization: Uses matplotlib to visualize clustering results and classifier predictions.

## Results

The script outputs:
- Visualization of K-means clustering results
- Accuracy and detailed classification reports for both Random Forest and Naive Bayes classifiers
- Visualizations of predictions from both classifiers on the first image

## Future Improvements

As this is a first project, there are several areas for potential improvement:
- Experiment with different numbers of clusters in K-means
- Try other supervised learning algorithms
- Implement cross-validation for more robust evaluation
- Explore feature engineering to improve classification accuracy

## Note

This project is part of a learning process. The code and approach may not be optimal or suitable for production use.
