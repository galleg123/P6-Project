import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

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
df = pd.read_csv('featuresExtracted.csv', usecols=columns)

# Make a new cage column that applies the function to each row
df['Cage'] = df.apply (lambda row: cage_detection(row), axis=1)

# Check for duplicate frames, i.e. where both a cage and a person is within the frame
subset_cols = ['Frame','BoudingBox','Area','Circularity','Convexity','Rectangularity','Elongation','Eccentricity','Solidity']
df = df.drop_duplicates(subset=subset_cols, keep='first')

# Save the new csv file with no doubles and a ['Cage'] column
df.to_csv('featuresExtracted_noDoubleFrames.csv',index=False)

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


# Write down test attempts for the different parameter (I found these specific ones online)
param_grid = [
	{'C': [0.5,1,10,100],
	'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
	'kernel': ['rbf']},
]

# Do a cross validation using GridSearchCV
optimal_params = GridSearchCV(
	SVC(),
	param_grid,
	cv=5,
	scoring='accuracy',
	verbose=1
)

# Fit the parameters using GridSearchCV
optimal_params.fit(X_train_scaled, y_train)

# Print the optimal parameters
print(optimal_params.best_params_)

# Try making a SVC with the new parameters provided by the CV
clf_svm = SVC(random_state=42, C=10, gamma=1)
clf_svm.fit(X_train_scaled, y_train)

predictions = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, predictions, labels=clf_svm.classes_)

# Display new confusion matrix
after = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Is not a cage","Is a cage"])
after.plot()
plt.savefig('after_CrossValidation.pdf')
