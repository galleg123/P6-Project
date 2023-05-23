import pandas as pd
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


# Creating a function that uses category_id to determine if a cage is present.
def cage_detection(row):
    if (row['category_id'] == 0) or (row['category_id'] == 1):
        return 1
    else:
        return 0


# Defining the different columns and features
columns = ['video_id', 'path', 'width', 'height', 'fps', 'total_frames', 'file_name', 'category_id', 'category_name',
           'supercategory', 'color', 'metadata', 'Frame', 'BoudingBox', 'Area', 'Circularity', 'Convexity',
           'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']
features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

# Import csv with all data
df = pd.read_csv('featureExtracted_noEdge_noDoubtleFrames.csv', usecols=columns)

# Make a new cage column that applies the function to each row
df['Cage'] = df.apply(lambda row: cage_detection(row), axis=1)

# Check for duplicate frames, i.e. where both a cage and a person is within the frame
subset_cols = ['Frame', 'BoudingBox', 'Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation',
               'Eccentricity', 'Solidity']
df = df.drop_duplicates(subset=subset_cols, keep='first')

# Save the new csv file with no doubles and a ['Cage'] column
df.to_csv('featuresExtracted_final.csv', index=False)

# Make a dataframe containing all features
X = df[features]

# Make a column with results of the classification
y = df['Cage']

# ----------------------------------------------------------------------------------------------------------------------
# KNN with mahalanobis

# Prepare test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale test data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Calculate the covariance matrix
covariance_matrix = EmpiricalCovariance().fit(X_train_scaled)

print("X_train shape:", X_train_scaled.shape)
print("Covariance shape:", covariance_matrix.covariance_.shape)

# Define the KNN classifier with Mahalanobis distance
def mahalanobis_distance(X, Z):
    inv_covariance = np.linalg.inv(covariance_matrix.covariance_)
    distances = distance.mahalanobis(X, Z, inv_covariance)
    return distances


# Create a KNN classifier object
knn = KNeighborsClassifier(n_neighbors=3,
                           metric=mahalanobis_distance, n_jobs=-1)  # You can choose the number of neighbors (K) here

# Train the KNN classifier on the training data
knn.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test_scaled)

# Construct a confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)

# This plots the confusion matrix
before = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
before.plot()
plt.savefig('knn_mahalanobis_rapoVr.pdf')

knn_accuracy = knn.score(X_train_scaled, y_train)
print("KNN Accuracy:", knn_accuracy)