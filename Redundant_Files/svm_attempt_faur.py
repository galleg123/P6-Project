import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# Creating a function that uses category_id to determine if a cage is present.
def cage_detection(row):
	if (row['category_id'] == 0) or (row['category_id'] == 1):
		return 1
	else:
		return 0
		
# Defining the different columns and features
columns = ['video_id','path','width','height','fps','total_frames','file_name','category_id','category_name','supercategory','color','metadata','Frame','BoudingBox','Area','Circularity','Convexity','Rectangularity','Elongation','Eccentricity','Solidity']
features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

# Import csv with all data
df = pd.read_csv('featureExtracted_noEdge_noDoubleFrames.csv', usecols=columns)

# Make a new cage column that applies the function to each row
df['Cage'] = df.apply (lambda row: cage_detection(row), axis=1)

# Make a dataframe containing all features
X = df[features]

# Make a column with results of the classification
y = df['Cage']

# Prepare test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale test data
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

"""
# Create a Support Vector Classifier (SVC) with preset parameters
clf_svm = SVC(random_state=42)

# Train the Support vector classifier
clf_svm.fit(X_train_scaled, y_train)


# Predict using the standard SVC
predictions = clf_svm.predict(X_test_scaled)

# Construct a confusion matrix
cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)

# This plots the confusion matrix
before = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Is not a cage","Is a cage"])
before.plot()
plt.savefig('before_CrossValidation.pdf')
plt.clf()
"""
# Write down test attempts for the different parameter (I found these specific ones online)
param_grid = [
	{'C': [1],
	'gamma': ['scale', 1],
	'kernel': ['rbf'],
	'class_weight':[{0: 1, 1:w} for w in [1, 2, 3, 4, 5]]},

]

from sklearn.metrics import make_scorer,fbeta_score
def f2_func(y_true, y_pred):
    f2_score = fbeta_score(y_true, y_pred, beta=5)
    return f2_score

def my_f2_scorer():
    return make_scorer(f2_func)
# Do a cross validation using GridSearchCV
optimal_params = GridSearchCV(
	SVC(),
	param_grid,
	cv=5,
	scoring=my_f2_scorer(),
	verbose=1,
	n_jobs=-1
)

# Fit the parameters using GridSearchCV
optimal_params.fit(X_train_scaled, y_train)

# Print the optimal parameters
print(optimal_params.best_params_)

# Try making a SVC with the new parameters provided by the CV
clf_svm = SVC(random_state=42, C=optimal_params.best_params_['C'], gamma=optimal_params.best_params_['gamma'], kernel=optimal_params.best_params_['kernel'], class_weight=optimal_params.best_params_['class_weight'])
#clf_svm = SVC(random_state=42, C=, gamma=1, kernel='rbf')
clf_svm.fit(X_train_scaled, y_train)

support_vectors = clf_svm.support_vectors_
mean_vector = np.mean(support_vectors, axis=0)
covariance_matrix = np.cov(support_vectors.T)

mahalanobis_distances = []
for sv in support_vectors:
    distance = mahalanobis(sv, mean_vector, np.linalg.inv(covariance_matrix))
    mahalanobis_distances.append(distance)

plt.plot(mahalanobis_distances)
plt.xlabel('Support Vector Index')
plt.ylabel('Mahalanobis Distance')
plt.savefig('mahalanobis_distances.pdf')
plt.clf()
predictions = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)

# Display new confusion matrix
after = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Is not a cage","Is a cage"])
after.plot()
plt.savefig('after_CrossValidation.pdf')
plt.clf()