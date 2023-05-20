import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score, make_scorer
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.tree import DecisionTreeClassifier


# Creating a function that uses category_id to determine if a cage is present.
def cage_detection(row):
    if (row['category_id'] == 0) or (row['category_id'] == 1):
        return 1
    else:
        return 0


# Defining the different columns and features
columns = ['video_id', 'path', 'width', 'height', 'fps', 'total_frames', 'file_name', 'category_id', 'category_name',
           'supercategory', 'color', 'metadata', 'pass_id', 'Frame', 'BoudingBox', 'Area', 'Circularity', 'Convexity',
           'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']
features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

# Import csv with all data
df = pd.read_csv('featuresExtracted_no_touchy_noDoubleFrames.csv', usecols=columns)

# Make a new cage column that applies the function to each row
df['Cage'] = df.apply(lambda row: cage_detection(row), axis=1)

# Make a dataframe containing all features
X = df[features]

# Make a column with results of the classification
y = df['Cage']

# Prepare test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale test data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Create a KNN classifier object
knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the number of neighbors (K) here
# Bagging ensemble
bagging_model = BaggingClassifier(knn, n_estimators=15)  # n_estimators=14 is best
# Create an AdaBoostClassifier with the base weighted KNN classifier
#adaboost = AdaBoostClassifier(base_estimator=knn, n_estimators=11)

# Train the KNN classifier on the training data
knn.fit(X_train_scaled, y_train)
bagging_model.fit(X_train_scaled, y_train)
# adaboost.fit(X_train_scaled, y_train) #does not work

# Predict using the KNN
predictions = knn.predict(X_test_scaled)
predictions_bagged = bagging_model.predict(X_test_scaled)
# predictions_boosted = adaboost.predict(X_test_scaled) #does not work

# Construct a confusion matrix
cm = confusion_matrix(y_test, predictions, labels=knn.classes_)
cm_bagged = confusion_matrix(y_test, predictions_bagged, labels=bagging_model.classes_)
# cm_boosted = confusion_matrix(y_test, predictions_boosted, labels=adaboost.classes_) #does not work

# This plots the confusion matrix
before = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
before.plot()
plt.savefig('knn_before_CrossValidation.pdf')
plt.savefig('knn_before_CrossValidation.png')
plt.clf()

# This plots the confusion matrix
before_bagged = ConfusionMatrixDisplay(confusion_matrix=cm_bagged, display_labels=["Is not a cage", "Is a cage"])
before.plot()
plt.savefig('knn_before_bagged_CrossValidation.pdf')
plt.savefig('knn_before_bagged_CrossValidation.png')
plt.clf()

# This plots the confusion matrix, does not work
# before_boosted = ConfusionMatrixDisplay(confusion_matrix=cm_boosted, display_labels=["Is not a cage", "Is a cage"])
# before.plot()
# plt.savefig('knn_before_boosted_CrossValidation.pdf')
# plt.savefig('knn_before_boosted_CrossValidation.png')
# plt.clf()

knn_accuracy = knn.score(X_train_scaled, y_train)
knn_bagged_accuracy = bagging_model.score(X_train_scaled, y_train)
# knn_boosted_accuracy = adaboost.score(X_train_scaled, y_train)
print("KNN before Cross Validation Accuracy:", knn_accuracy)
print("KNN bagged before Cross Validation Accuracy:", knn_bagged_accuracy)
# print("KNN boosted before Cross Validation Accuracy:", knn_boosted_accuracy)

# ----------------------------------------------------------------------------------------------------------------------

# Define the parameter grid for grid search
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

f2_scorer = make_scorer(fbeta_score, beta=2)
f4_scorer = make_scorer(fbeta_score, beta=4)

# Do a cross validation using GridSearchCV
optimal_params = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring=f4_scorer,
    refit=True,
    verbose=1,
    n_jobs=-1
)

# Fit the parameters using GridSearchCV
optimal_params.fit(X_train_scaled, y_train)

Results = pd.DataFrame(optimal_params.cv_results_).to_csv("knn_GridSearchResults.csv")

# Print the best parameters and best score
print("Best KNN Parameters: ", optimal_params.best_params_)
print("Best KNN Score: ", optimal_params.best_score_)

# Uses the best estimator from grid search to predict test data
knn = optimal_params.best_estimator_

predictions = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=knn.classes_)

# Display new confusion matrix
after = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Is not a cage", "Is a cage"])
after.plot()
plt.savefig('knn_after_CrossValidation.pdf')
plt.savefig('knn_after_CrossValidation.png')
plt.clf()

knn_accuracy = knn.score(X_train_scaled, y_train)
print("KNN after Cross Validation Accuracy:", knn_accuracy)

# ----------------------------------------------------------------------------------------------------------------------
bagging_model = BaggingClassifier(knn, n_estimators=14)  # n_estimators=14 is best

# Define the parameter grid for grid search
param_grid_bagging = {
    'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
}

# Do a cross validation using GridSearchCV
optimal_params_bagging = GridSearchCV(
    BaggingClassifier(),
    param_grid_bagging,
    cv=5,
    scoring=f4_scorer,
    refit=True,
    verbose=1,
    n_jobs=-1
)

# Fit the parameters using GridSearchCV
optimal_params_bagging.fit(X_train_scaled, y_train)

Results_bagging = pd.DataFrame(optimal_params_bagging.cv_results_).to_csv("knn_bagging_GridSearchResults.csv")

# Print the best parameters and best score
print("Best KNN bagged Parameters: ", optimal_params_bagging.best_params_)
print("Best KNN bagged Score: ", optimal_params_bagging.best_score_)

# Uses the best estimator from grid search to predict test data
knn_bagged = optimal_params.best_estimator_

predictions_bagged = knn_bagged.predict(X_test_scaled)
cm_bagged = confusion_matrix(y_test, predictions_bagged, labels=knn_bagged.classes_)

# Display new confusion matrix
after_bagged = ConfusionMatrixDisplay(confusion_matrix=cm_bagged, display_labels=["Is not a cage", "Is a cage"])
after.plot()
plt.savefig('knn_bagged_after_CrossValidation.pdf')
plt.savefig('knn_bagged_after_CrossValidation.png')
plt.clf()

knn_accuracy_bagged = knn_bagged.score(X_train_scaled, y_train)
print("KNN bagged after Cross Validation Accuracy:", knn_accuracy_bagged)

