import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, fbeta_score, make_scorer
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
plt.savefig('before_CrossValidation.png')
plt.clf()

# Write down test attempts for the different parameter (I found these specific ones online)
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'linear'], 'class_weight': [None, "balanced", {0:1, 1:3}]}
#param_grid = {'C': [10], 'gamma': [1],'kernel': ['rbf'], 'class_weight': [None, "balanced", {0:1, 1:3}]}

f2_scorer = make_scorer(fbeta_score, beta=2)
f4_scorer = make_scorer(fbeta_score, beta=4)

# Do a cross validation using GridSearchCV
optimal_params = GridSearchCV(
	SVC(),
	param_grid,
	cv=5,
	scoring=f2_scorer,
	refit=True,
	verbose=1,
	n_jobs=6
)

# Fit the parameters using GridSearchCV
optimal_params.fit(X_train_scaled, y_train)

# Print the optimal parameters
print(optimal_params.best_params_)

'''
# Try making a SVC with the new parameters provided by the CV
clf_svm = SVC(random_state=42, C=optimal_params.best_params_['C'], gamma=optimal_params.best_params_['gamma'], kernel=optimal_params.best_params_['kernel'])
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
'''
clf_svm = optimal_params.best_estimator_
predictions = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)

# Display new confusion matrix
after = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Is not a cage","Is a cage"])
after.plot()
plt.savefig('after_CrossValidation.pdf')
plt.savefig('after_CrossValidation.png')
plt.clf()

