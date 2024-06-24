import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# Function to load satellite image data
def load_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read().transpose(1, 2, 0)  # Transpose to (height, width, channels)
    return image


# Function to prepare dataset
def prepare_dataset(images):
    X = np.vstack([img.reshape(-1, img.shape[-1]) for img in images])
    return X


# Function to visualize labels on an image
def visualize_labels(image, labels, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(labels, alpha=0.5, cmap='jet')  # Use a colorful colormap
    plt.colorbar(label='Class')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Load satellite images
image_files = ['Data/1.jpg', 'Data/2.jpg', 'Data/3.jpg', 'Data/4.jpg', 'Data/5.jpg', 'Data/6.jpg', 'Data/7.jpg']
images = [load_image(file) for file in image_files]

# Prepare dataset
X = prepare_dataset(images)

# Perform K-means clustering
n_clusters = 5  # You can adjust this based on expected land cover classes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# Reshape labels back to image shape
label_images = [
    labels[i * images[0].shape[0] * images[0].shape[1]:(i + 1) * images[0].shape[0] * images[0].shape[1]].reshape(
        images[0].shape[:2])
    for i in range(len(images))]

# Visualize labels for the first image
visualize_labels(images[0], label_images[0], 'K-means Clustering Results')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate Maximum Likelihood classifier (Random Forest)
max_likelihood_clf = RandomForestClassifier(random_state=42)
max_likelihood_clf.fit(X_train, y_train)
max_likelihood_preds = max_likelihood_clf.predict(X_test)

print("Maximum Likelihood Classifier (Random Forest):")
print("Accuracy:", accuracy_score(y_test, max_likelihood_preds))
print("Classification Report:")
print(classification_report(y_test, max_likelihood_preds))

# Train and evaluate Naïve Bayes classifier
naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(X_train, y_train)
naive_bayes_preds = naive_bayes_clf.predict(X_test)

print("\nNaïve Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, naive_bayes_preds))
print("Classification Report:")
print(classification_report(y_test, naive_bayes_preds))

# Visualize predictions for the first image
ml_pred_image = max_likelihood_clf.predict(images[0].reshape(-1, images[0].shape[-1])).reshape(images[0].shape[:2])
visualize_labels(images[0], ml_pred_image, 'Maximum Likelihood (Random Forest) Predictions')

nb_pred_image = naive_bayes_clf.predict(images[0].reshape(-1, images[0].shape[-1])).reshape(images[0].shape[:2])
visualize_labels(images[0], nb_pred_image, 'Naïve Bayes Predictions')
